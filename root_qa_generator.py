import json
import os
import random
import time
from pathlib import Path

class OptimizedQAGenerator:
    """优化后的问答对生成器（忽略构造函数）"""
    
    def __init__(self, analysis_file: str):
        self.analysis_file = analysis_file
        self.analysis_data = self._load_analysis_data()
        self.class_data = self._extract_class_info()
    
    def _load_analysis_data(self) -> dict:
        """加载代码分析结果"""
        if not os.path.exists(self.analysis_file):
            raise FileNotFoundError(f"分析文件 {self.analysis_file} 不存在")
        
        with open(self.analysis_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _extract_class_info(self) -> dict:
        """提取类信息"""
        class_map = {}
        
        # 提取所有类信息
        for class_info in self.analysis_data.get('class_architectures', []):
            class_name = class_info.get('name', '')
            if class_name:
                class_map[class_name] = class_info
        
        return class_map
    
    def _filter_constructor_methods(self, methods: list, class_name: str) -> list:
        """过滤掉构造函数和析构函数"""
        filtered = []
        
        for method in methods:
            method_name = method.get("name", "")
            
            # 跳过构造函数 (与类名相同)
            if method_name == class_name:
                continue
                
            # 跳过析构函数 (以 ~ 开头后跟类名)
            if method_name.startswith("~") and method_name[1:] == class_name:
                continue
                
            # 跳过复制构造函数和移动构造函数 (如果有类似命名)
            if method_name == "operator=":
                continue
                
            # 保留其他方法
            filtered.append(method)
            
        return filtered
    
    def generate_qa_pairs(self, count: int = 50) -> list:
        """生成指定数量的问答对"""
        qa_pairs = []
        classes = list(self.class_data.keys())
        
        # 确保有足够的类来生成问答对
        if not classes:
            print("警告: 源码分析结果中没有提取到类信息")
            return qa_pairs
        
        # 生成问题类型分布权重
        qa_types = [
            "class_location", 
            "base_classes",
            "methods_list",
            "method_detail",
        ]
        weights = [0.25, 0.25, 0.25, 0.25]  # 调整问题类型分布
        
        for _ in range(count):
            # 随机选择一个类
            class_name = random.choice(classes)
            class_info = self.class_data[class_name]
            
            # 根据权重随机选择问题类型
            qa_type = random.choices(qa_types, weights)[0]
            
            # 根据类型生成问答对
            if qa_type == "class_location":
                qa_pairs.append(self._generate_location_qa(class_info))
            elif qa_type == "base_classes":
                qa_pair = self._generate_base_classes_qa(class_info)
                if qa_pair:  # 只有类有基类时才添加
                    qa_pairs.append(qa_pair)
            elif qa_type == "methods_list":
                qa_pair = self._generate_methods_qa(class_info)
                if qa_pair:  # 只有类有方法时才添加
                    qa_pairs.append(qa_pair)
            elif qa_type == "method_detail":
                qa_pair = self._generate_method_detail_qa(class_info)
                if qa_pair:  # 只有类有方法时才添加
                    qa_pairs.append(qa_pair)
            '''
            else:
                qa_pair = self._generate_field_qa(class_info)
                if qa_pair:  # 只有类有字段时才添加
                    qa_pairs.append(qa_pair)
            '''
        
        return qa_pairs[:count]  # 确保不超过请求数量
    
    def _generate_location_qa(self, class_info: dict) -> dict:
        """生成类位置问答对"""
        class_name = class_info["name"]
        file_path = class_info.get("file_path", "未知位置")
        
        return {
            "question": f"{class_name}类定义在哪个文件中？",
            "answer": f"{class_name}类定义在{file_path}文件中",
            "metadata": {
                "type": "class_location",
                "class": class_name,
                "source": "code_analysis"
            }
        }
    
    def _generate_base_classes_qa(self, class_info: dict) -> dict or None:
        """生成基类关系问答对"""
        class_name = class_info["name"]
        base_classes = class_info.get("base_classes", [])
        
        # 如果类没有基类，则不生成问题
        if not base_classes:
            return None
            
        base_str = ", ".join(base_classes)
        
        return {
            "question": f"{class_name}类继承自哪些基类？",
            "answer": f"{class_name}类继承自: {base_str}",
            "metadata": {
                "type": "base_classes",
                "class": class_name,
                "source": "code_analysis"
            }
        }
    
    def _generate_methods_qa(self, class_info: dict) -> dict or None:
        """生成方法列表问答对（忽略构造函数）"""
        class_name = class_info["name"]
        methods = class_info.get("methods", [])
        
        # 过滤掉构造函数和析构函数
        filtered_methods = self._filter_constructor_methods(methods, class_name)
        
        # 如果没有实际方法，则不生成问题
        if not filtered_methods:
            return None
        
        # 从分析结果提取方法名（忽略构造函数）
        method_names = []
        for method in filtered_methods:
            method_name = method.get("name", "")
            if method_name:
                method_names.append(method_name)
        
        if not method_names:
            return None
            
        # 最多展示5个方法
        methods_str = ", ".join(method_names[:5])
        answer = f"{class_name}类包含以下重要方法: {methods_str}"
        
        # 增加额外提示如果方法超过5个
        if len(method_names) > 5:
            answer += f" 等{len(method_names)}个方法"
        
        return {
            "question": f"{class_name}类包含哪些重要方法（排除构造函数）？",
            "answer": answer,
            "metadata": {
                "type": "key_methods",
                "class": class_name,
                "source": "code_analysis"
            }
        }
    
    def _generate_method_detail_qa(self, class_info: dict) -> dict or None:
        """生成方法详情问答对（忽略构造函数）"""
        class_name = class_info["name"]
        methods = class_info.get("methods", [])
        
        # 过滤掉构造函数和析构函数
        filtered_methods = self._filter_constructor_methods(methods, class_name)
        
        # 如果没有实际方法，则不生成问题
        if not filtered_methods:
            return None
        
        # 从过滤后的方法中随机选择一个
        method = random.choice(filtered_methods)
        method_name = method.get("name", "")
        
        if not method_name:
            return None
        
        # 从分析结果提取方法详情
        params = method.get("parameters", [])
        return_type = method.get("return_type", "void")
        
        # 构建参数列表
        param_list = []
        for param in params:
            param_type = param.get("type", "unknown")
            param_name = param.get("name", "")
            param_list.append(f"{param_type} {param_name}")
        
        # 如果有参数则显示，否则显示void
        param_str = ", ".join(param_list) if param_list else "void"
        
        return {
            "question": f"{class_name}类的{method_name}方法的签名是什么？",
            "answer": f"函数签名: {return_type} {method_name}({param_str})",
            "metadata": {
                "type": "method_detail",
                "class": class_name,
                "method": method_name,
                "source": "code_analysis"
            }
        }
    
    def _generate_field_qa(self, class_info: dict) -> dict or None:
        """生成字段信息问答对"""
        class_name = class_info["name"]
        fields = class_info.get("fields", [])
        
        # 如果没有字段，则不生成问题
        if not fields:
            return None
        
        # 随机选择一个字段
        field = random.choice(fields)
        field_name = field.get("name", "")
        field_type = field.get("type", "未知类型")
        
        if not field_name:
            return None
        
        # 根据字段名称猜测用途
        purpose = "存储类的重要数据"
        if field_name.startswith("f"):
            base_name = field_name[1:]
            if base_name.lower() in ["entries", "nevents"]:
                purpose = "记录条目数量"
            elif base_name.lower() in ["size", "length", "n"]:
                purpose = "表示数据结构的大小"
            elif base_name.lower().endswith("array") or base_name.lower().endswith("list"):
                purpose = "保存数据集合"
            elif base_name.lower() in ["pointer", "ptr", "ref"]:
                purpose = "引用资源或对象"
        
        return {
            "question": f"{class_name}类的{field_name}字段的用途是什么？",
            "answer": f"{field_name}是一个{field_type}类型的字段，主要用于{purpose}",
            "metadata": {
                "type": "field_info",
                "class": class_name,
                "field": field_name,
                "source": "code_analysis"
            }
        }
    
    def save_qa(self, qa_pairs: list, output_file: str):
        """保存问答对到文件"""
        dataset = {
            "name": "ROOT_QA_Dataset",
            "version": "1.0",
            "source_file": self.analysis_file,
            "qa_count": len(qa_pairs),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "qa_pairs": qa_pairs
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

# 使用示例
if __name__ == "__main__":
    # 输入文件：ROOT源码分析结果
    ANALYSIS_FILE = "root_analysis.json"
    # 输出文件：问答对数据集
    OUTPUT_FILE = "root_qa_pairs.json"
    
    try:
        print("开始生成基于ROOT源码分析的问答对...")
        start_time = time.time()
        
        # 创建生成器并生成50个问答对
        generator = OptimizedQAGenerator(ANALYSIS_FILE)
        qa_pairs = generator.generate_qa_pairs(5000)
        
        # 保存结果
        generator.save_qa(qa_pairs, OUTPUT_FILE)
        
        elapsed = time.time() - start_time
        print(f"成功生成 {len(qa_pairs)} 个基于源码分析的问答对，耗时: {elapsed:.2f}秒")
        print(f"结果已保存到 {OUTPUT_FILE}")
        
        # 打印一些示例
        if qa_pairs:
            print("\n示例问答对:")
            for i, qa in enumerate(qa_pairs[:3]):
                print(f"{i+1}. 问题: {qa['question']}")
                print(f"   答案: {qa['answer']}")
                print(f"   元数据: {str(qa['metadata'])[:80]}...")
        
    except Exception as e:
        print(f"生成问答对时出错: {e}")