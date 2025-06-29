# CERNROOTSourceAnalyst
 用于分析CERN ROOT框架源代码并生成机器学习训练数据集。 此项目为一个测试项目。

## 核心功能：

- 自动化问答对生成
- 智能设计方案生成

# 源码解析阶段

使用tree-sitter对ROOT源码进行深度分析，提取关键结构信息： 

## 目标
- 识别所有类定义（包括成员方法、字段、基类）
- 提取完整类继承关系
- 解析方法签名
- 生成结构化 JSON 数据存储解析结果


# 训练阶段

基于Qwen 2.0，使用结构化的JSON数据存储解析结果数据进行训练，目前已经训练了一个初步的模型。

# 安装与使用

1. 获取ROOT源码  
`git clone https://github.com/root-project/root.git` 
2. 安装Python的依赖  
`pip install -r dependencies.txt`
3. 执行源码分析脚本  
`python root_analysis.py /path-to-root/`  
其中/path-to-root/代表root源码的路径
4. 执行产生格式化数据的脚本  
`python root_qa_generator.py`
5. 进入train文件夹，执行训练和测试（需要Qwen2.0-7b模型），且需要在脚本qwen_finetune_tool.py中指定模型的位置  
    - 训练：`python qwen_finetune_tool.py --train --qa_data root_qa_pairs.json --epochs 5 --batch_size 2 --grad_accum_steps 4`
    - 测试：`python qwen_finetune_tool.py --test --adapter_path ./output/adapter --test_data test_qa_pairs.json`
