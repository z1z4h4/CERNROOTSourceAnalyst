
"""
任务：Qwen2-7B 问答对微调一体化工具，集成数据准备、模型微调、性能评估和API部署功能
时间：2025年6月18日
作者：z1z4h4
"""

import os
import json
import torch
import argparse
import warnings
from tqdm import tqdm
from typing import List, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import datasets
import gradio as gr

# 忽略不必要的警告
warnings.filterwarnings("ignore")
datasets.disable_progress_bar()

def load_qa_data(data_path: str) -> List[Dict[str, Any]]:
    """更健壮的数据加载函数"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 检查常见数据结构
    if isinstance(data, dict):
        # 如果是字典格式，尝试提取问答对列表
        if "qa_pairs" in data:
            return data["qa_pairs"]
        elif "data" in data:
            return data["data"]
        else:
            return [v for v in data.values() if isinstance(v, list)]
    
    # 如果是列表但元素格式不统一
    if isinstance(data, list):
        valid_data = []
        for item in data:
            if isinstance(item, dict):
                valid_data.append(item)
            elif isinstance(item, list):
                valid_data.extend(item)
        return valid_data
    
    raise ValueError(f"无法识别的数据格式: {type(data)}")

def format_for_qwen(qa_data: List[Any]) -> List[Dict[str, Any]]:
    """更安全的格式转换函数"""
    formatted = []
    for item in qa_data:
        # 处理各种可能的数据格式
        if isinstance(item, dict):
            # 标准问答对格式
            if "question" in item and "answer" in item:
                formatted.append({
                    "conversations": [
                        {"role": "user", "content": item["question"]},
                        {"role": "assistant", "content": item["answer"]}
                    ]
                })
            # 其他可能的键名变体
            elif "q" in item and "a" in item:
                formatted.append({
                    "conversations": [
                        {"role": "user", "content": item["q"]},
                        {"role": "assistant", "content": item["a"]}
                    ]
                })
            # 处理嵌套结构
            elif "data" in item and isinstance(item["data"], list):
                formatted.extend(format_for_qwen(item["data"]))
        
        # 处理元组形式 (问题, 答案)
        elif isinstance(item, tuple) and len(item) == 2:
            formatted.append({
                "conversations": [
                    {"role": "user", "content": item[0]},
                    {"role": "assistant", "content": item[1]}
                ]
            })
        
        # 处理列表形式 [问题, 答案]
        elif isinstance(item, list) and len(item) == 2:
            formatted.append({
                "conversations": [
                    {"role": "user", "content": item[0]},
                    {"role": "assistant", "content": item[1]}
                ]
            })
    
    return formatted

def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
    """数据集分词处理"""
    def tokenize_function(examples):
        texts = [
            tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in examples["conversations"]
        ]
        return tokenizer(
            texts, 
            padding="max_length", 
            truncation=True, 
            max_length=1024,
            return_tensors="pt"
        )
    
    return dataset.map(tokenize_function, batched=True, batch_size=100)

def train_model(args):
    """模型微调主函数"""
    # 1. 加载并处理数据
    print(f"加载问答对数据: {args.qa_data}")
    qa_data = load_qa_data(args.qa_data)
    print(f"加载成功! 共 {len(qa_data)} 个问答对")
    
    formatted_data = format_for_qwen(qa_data)
    dataset = Dataset.from_list(formatted_data)
    print("数据格式化完成")
    
    # 2. 加载模型和分词器
    print(f"加载模型: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, 
        trust_remote_code=True,
        pad_token="<|endoftext|>",
        padding_side="right"
    )
    
    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    print("模型加载完成!")
    
    # 3. 准备训练和LoRA
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 4. 准备数据集
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    print("数据集预处理完成")
    
    # 5. 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=200,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none",
        fp16=True,
        optim="paged_adamw_32bit",
        warmup_ratio=0.1
    )
    
    # 6. 创建并启动训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("开始训练...")
    trainer.train()
    print(f"训练完成! 结果保存至 {args.output_dir}")
    
    # 7. 保存适配器
    adapter_path = os.path.join(args.output_dir, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"适配器已保存至 {adapter_path}")
    
    return adapter_path

def setup_inference(adapter_path, base_model):
    """设置推理环境"""
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True
    )
    
    try:
        # 尝试直接加载合并模型
        model = AutoModelForCausalLM.from_pretrained(
            adapter_path,
            trust_remote_code=True,
            #device_map="auto",
            torch_dtype=torch.bfloat16
        )
        print("加载合并模型成功")
    except (OSError, ValueError) as e:
        print(f"合并模型加载失败: {e}, 尝试加载适配器模式...")
        # 分别加载基础模型和适配器
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        print("加载适配器成功")
    
    # 创建推理管道 (不指定device参数)
    qa_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    
    # 定义响应函数
    def generate_response(message, history=None):
        """生成模型响应，支持单参数调用"""
        prompt = f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
        try:
            response = qa_pipeline(
                prompt,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )[0]['generated_text']
            return response.split("<|im_start|>assistant\n")[-1].strip()
        except Exception as e:
            return f"生成回答时出错: {str(e)}"
    
    return generate_response

def evaluate_model(chatbot_fn, test_data, tokenizer, num_samples=50):
    """
    评估微调后的模型性能
    :param chatbot_fn: 模型生成函数
    :param test_data: 测试数据集
    :param tokenizer: 分词器
    :param num_samples: 抽样数量
    """
    # 随机选择测试样本
    if len(test_data) > num_samples:
        test_samples = random.sample(test_data, num_samples)
    else:
        test_samples = test_data
        
    results = {
        'correct': 0,
        'partially_correct': 0,
        'incorrect': 0,
        'similarity_scores': []
    }
    
    for i, item in enumerate(test_samples):
        # 确保测试数据格式正确
        if 'question' not in item or 'answer' not in item:
            continue
            
        question = item['question']
        expected = item['answer']
        
        # 获取模型响应
        try:
            response = chatbot_fn(question)
            
            # 跳过错误信息
            if "生成回答时出错" in response:
                print(f"[{i+1}/{len(test_samples)}] 问题: '{question[:40]}...' 生成错误")
                results['incorrect'] += 1
                results['similarity_scores'].append(0.0)
                continue
            
            # 文本相似度计算
            expected_tokens = tokenizer.encode(expected, add_special_tokens=False)
            response_tokens = tokenizer.encode(response, add_special_tokens=False)
            
            # 创建向量表示（简单长度归一化）
            expected_vec = np.zeros(tokenizer.vocab_size)
            response_vec = np.zeros(tokenizer.vocab_size)
            
            # 计数
            for token in expected_tokens:
                expected_vec[token] += 1
            for token in response_tokens:
                response_vec[token] += 1
                
            # 归一化
            if np.linalg.norm(expected_vec) > 0:
                expected_vec /= np.linalg.norm(expected_vec)
            if np.linalg.norm(response_vec) > 0:
                response_vec /= np.linalg.norm(response_vec)
            
            # 余弦相似度
            similarity = np.dot(expected_vec, response_vec) / (
                np.linalg.norm(expected_vec) * np.linalg.norm(response_vec) + 1e-8
            )
            
            results['similarity_scores'].append(similarity)
            
            # 判断正确性
            if similarity > 0.8:
                results['correct'] += 1
            elif similarity > 0.6:
                results['partially_correct'] += 1
            else:
                results['incorrect'] += 1
                
            # 打印前3个样本的详细情况
            if i < 3:
                print(f"\n样本 {i+1}:")
                print(f"问题: {question}")
                print(f"预期答案: {expected}")
                print(f"模型响应: {response}")
                print(f"相似度: {similarity:.4f}")
            elif i == 3:
                print("\n... 更多样本统计中 ...")
        except Exception as e:
            print(f"处理问题 '{question[:40]}...' 时出错: {e}")
            results['incorrect'] += 1
            results['similarity_scores'].append(0.0)
                
    # 计算平均相似度
    if results['similarity_scores']:
        results['avg_similarity'] = np.mean(results['similarity_scores'])
    else:
        results['avg_similarity'] = 0.0
    
    return results

def run_web_ui(chatbot_fn):
    """启动Gradio Web界面"""
    with gr.Blocks(title="ROOT框架专家助手") as demo:
        gr.Markdown("## ROOT框架专家助手 - Qwen2-7B微调版")
        gr.Markdown("可以向助手提出关于ROOT框架的任何技术问题")
        
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(label="您的问题")
        clear_btn = gr.ClearButton([msg, chatbot])
        
        def respond(message, chat_history):
            response = chatbot_fn(message)
            chat_history.append((message, response))
            return "", chat_history
        
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    print("\n启动Web服务... 在浏览器中访问 http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860)

def main():
    parser = argparse.ArgumentParser(description="Qwen2-7B 问答对微调工具")
    
    # 主要功能选项
    parser.add_argument("--train", action="store_true", help="执行微调训练")
    parser.add_argument("--inference", action="store_true", help="启动推理API")
    parser.add_argument("--web", action="store_true", help="启动Web界面")
    
    # 数据参数
    parser.add_argument("--qa_data", type=str, default="root_qa_pairs.json", 
                        help="问答对JSON文件路径")
    
    # 模型参数
    parser.add_argument("--base_model", type=str, default="./qwen27b",
                        help="基础模型名称或路径")
    parser.add_argument("--adapter_path", type=str, default="./output/adapter",
                        help="微调后适配器路径")
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="输出目录")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="每设备批大小")
    parser.add_argument("--grad_accum_steps", type=int, default=8, 
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="学习率")
    
    args = parser.parse_args()
    
    # 执行训练
    if args.train:
        adapter_path = train_model(args)
        args.adapter_path = adapter_path  # 更新适配器路径
    
    # 执行推理和测试
    if args.inference or args.web or args.test:
        if not os.path.exists(args.adapter_path):
            print(f"错误: 适配器路径不存在 {args.adapter_path}")
            print("请先运行 --train 进行训练")
            return
        
        print("加载推理模型...")
        chatbot_fn = setup_inference(args.adapter_path, args.base_model)
        
        # 加载分词器用于评估
        tokenizer = AutoTokenizer.from_pretrained(
            args.adapter_path,
            trust_remote_code=True
        )
        
        # 启动Web界面
        if args.web:
            run_web_ui(chatbot_fn)
        
        # 命令行推理
        elif args.inference:
            print("\nROOT专家助手已启动 (输入'exit'退出)")
            while True:
                try:
                    question = input("\n您的问题: ")
                    if question.lower() == "exit":
                        break
                    # 单参数调用
                    response = chatbot_fn(question)
                    print(f"\n助手: {response}")
                except KeyboardInterrupt:
                    print("\n程序已终止")
                    break
                except Exception as e:
                    print(f"处理请求时出错: {e}")

if __name__ == "__main__":
    main()