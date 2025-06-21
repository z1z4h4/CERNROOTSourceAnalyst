import os
import sys
import json
import time
import argparse
import traceback
import fnmatch
from pathlib import Path
from tree_sitter_language_pack import get_parser, get_language

class ROOTCodeAnalyzer:
    """使用 tree_sitter_language_pack 分析 ROOT 框架代码（相对路径版本）"""
    
    # 排除测试文件的关键词模式（不区分大小写）
    TEST_EXCLUDE_PATTERNS = ['*test*', '*Test*', '*testing*', '*Testing*', '*mock*', '*Mock*']
    
    def __init__(self, debug=False):
        """初始化分析器"""
        self.debug = debug
        self.repo_root = None  # 用于存储仓库根目录路径
        
        try:
            # 确保 C++ 语言支持已安装
            try:
                self.parser = get_parser("cpp")
            except Exception as e:
                if "Language not found" in str(e):
                    print("安装 C++ 语言支持...")
                    self.parser = get_parser("cpp")
                else:
                    raise
            
            # 获取语言对象用于高级查询
            try:
                self.language = get_language("cpp")
            except:
                self.language = None
                print("警告: 无法获取语言对象，高级查询功能不可用")
            
            print("Tree-sitter 解析器初始化成功")
        except Exception as e:
            print(f"初始化 Tree-sitter 解析器失败: {e}")
            self.parser = None
            self.language = None
    
    def analyze_repository(self, repo_path: str, output_file: str = "root_analysis.json"):
        """分析整个 ROOT 代码仓库（增强错误处理）"""
        if not self.parser:
            print("错误: 没有可用的解析器")
            return {}
        
        # 保存仓库根目录路径
        self.repo_root = Path(repo_path).resolve()
        print(f"开始分析 ROOT 仓库: {self.repo_root}")
        print(f"排除测试模式: {', '.join(self.TEST_EXCLUDE_PATTERNS)}")
        
        # 创建结果字典
        results = {
            "metadata": {
                "repo_path": str(self.repo_root),
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "analyzed_files": 0,
                "error_files": [],
                "excluded_files": 0,
                "excluded_dirs": 0
            },
            "business_rules": [],
            "class_architectures": [],
            "physics_constants": [],
            "root_macros": []
        }
        
        # 确定要分析的核心目录
        core_dirs = ['core', 'io', 'math', 'tmva', 'hist', 'tree', 'net']
        analyzed_dirs = []
        excluded_files = 0
        excluded_dirs = 0
        
        # 遍历仓库目录
        for core_dir in core_dirs:
            dir_path = self.repo_root / core_dir
            if not dir_path.is_dir():
                print(f"跳过不存在的目录: {dir_path}")
                continue
                
            print(f"分析目录: {dir_path}")
            
            # 记录原始目录数量
            orig_dir_count = len(list(dir_path.iterdir()))
            
            # 遍历所有 C++ 文件
            for file_path in dir_path.rglob('*.[ch]xx'):
                if not file_path.is_file():
                    continue
                
                # 计算文件相对路径
                relative_path = file_path.relative_to(self.repo_root)
                relative_path_str = str(relative_path)
                
                # 检查文件是否测试相关
                if self.is_test_related(relative_path_str):
                    if self.debug:
                        print(f"排除测试文件: {relative_path_str}")
                    excluded_files += 1
                    continue
                
                if self.debug:
                    print(f"分析文件: {relative_path_str}")
                
                try:
                    # 使用相对路径进行分析
                    file_results = self.analyze_file(relative_path_str, file_path)
                    
                    # 如果文件分析失败，记录错误但继续处理其他文件
                    if "error" in file_results:
                        results["metadata"]["error_files"].append({
                            "file": relative_path_str,
                            "error": file_results["error"]
                        })
                        continue
                    
                    # 合并结果
                    for key in ["business_rules", "class_architectures", 
                               "physics_constants", "root_macros"]:
                        results[key].extend(file_results.get(key, []))
                    
                    results["metadata"]["analyzed_files"] += 1
                except Exception as e:
                    error_msg = f"分析文件 {relative_path_str} 时出错: {e}"
                    print(error_msg)
                    if self.debug:
                        traceback.print_exc()
                    results["metadata"]["error_files"].append({
                        "file": relative_path_str,
                        "error": error_msg
                    })
            
            analyzed_dirs.append(str(dir_path.relative_to(self.repo_root)))
            
            # 统计排除的目录
            final_dir_count = len(list(dir_path.iterdir()))
            excluded_dirs += (orig_dir_count - final_dir_count)
        
        # 更新结果中的排除计数
        results["metadata"]["excluded_files"] = excluded_files
        results["metadata"]["excluded_dirs"] = excluded_dirs
        results["metadata"]["analyzed_dirs"] = analyzed_dirs
        results["metadata"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        print("\n" + "="*60)
        print("ROOT 代码分析摘要:")
        print("="*60)
        print(f"- 源文件目录: {self.repo_root}")
        print(f"- 分析目录: {', '.join(results['metadata']['analyzed_dirs'])}")
        print(f"- 分析文件数: {results['metadata']['analyzed_files']}")
        print(f"- 排除目录数: {results['metadata']['excluded_dirs']}")
        print(f"- 排除文件极 {results['metadata']['excluded_files']}")
        print(f"- 提取类: {len(results['class_architectures'])}")
        print(f"- 提取业务规则: {len(results['business_rules'])}")
        print(f"- 提取物理常量: {len(results['physics_constants'])}")
        print(f"- 提取ROOT宏: {len(results['root_macros'])}")
        if results["metadata"]["error_files"]:
            print(f"- 出错文件数: {len(results['metadata']['error_files'])}")
        print("="*60)
        print(f"分析完成! 结果已保存到 {output_file}")
        
        return results
    
    def is_test_related(self, path):
        """检查路径是否与测试相关（不区分大小写）"""
        path_lower = path.lower()
        return any(fnmatch.fnmatch(path_lower, pattern.lower()) 
                  for pattern in self.TEST_EXCLUDE_PATTERNS)
    
    def analyze_file(self, relative_path: str, absolute_path: Path = None) -> dict:
        """分析单个 C++ 文件（使用相对路径）"""
        if not self.parser:
            print(f"错误: 没有可用的解析器，跳过文件 {relative_path}")
            return {}
        
        # 如果有提供绝对路径则使用，否则从仓库根目录构建
        if absolute_path is None:
            absolute_path = self.repo_root / relative_path
            
        try:
            if self.debug:
                print(f"开始分析文件: {relative_path}")
            
            # 读取文件内容
            with open(absolute_path, 'rb') as f:
                source_bytes = f.read()
            
            # 解析文件
            tree = self.parser.parse(source_bytes)
            root_node = tree.root_node
            
            # 确保根节点有效
            if root_node is None:
                error_msg = f"文件 {relative_path} 解析失败，根节点为空"
                print(error_msg)
                return {
                    "file_path": relative_path,
                    "error": error_msg
                }
            
            if self.debug:
                print(f"文件 {relative_path} 解析成功，根节点类型: {root_node.type}")
            
            # 提取各种信息 - 使用相对路径
            business_rules = self.extract_business_rules(root_node, source_bytes, relative_path)
            class_architectures = self.extract_class_architectures(root_node, source_bytes, relative_path)
            physics_constants = self.extract_physics_constants(root_node, source_bytes, relative_path)
            root_macros = self.extract_root_macros(root_node, source_bytes, relative_path)
            
            if self.debug:
                print(f"文件 {relative_path} 分析完成:")
                print(f"  业务规则: {len(business_rules)}")
                print(f"  类定义: {len(class_architectures)}")
                print(f"  物理常量: {len(physics_constants)}")
                print(f"  ROOT 宏: {len(root_macros)}")
            
            return {
                "file_path": relative_path,
                "business_rules": business_rules,
                "class_architectures": class_architectures,
                "physics_constants": physics_constants,
                "root_macros": root_macros,
            }
        except Exception as e:
            error_msg = f"分析文件 {relative_path} 时出错: {e}"
            print(error_msg)
            if self.debug:
                traceback.print_exc()
            return {
                "file_path": relative_path,
                "error": error_msg
            }
    
    def extract_business_rules(self, root_node, source_bytes, relative_path):
        """提取业务规则并添加相对路径"""
        rules = []
        
        # 查找所有条件语句
        condition_nodes = self.find_nodes_by_type(root_node, "if_statement")
        for node in condition_nodes:
            if node is None:
                continue
            condition_text = self.get_node_text(node, source_bytes)
            rules.append({
                "type": "condition",
                "code": condition_text,
                "location": self.get_location(node),
                "file_path": relative_path  # 使用相对路径
            })
        
        # 查找所有循环语句
        loop_types = ["for_statement", "while_statement", "do_statement"]
        for loop_type in loop_types:
            loop_nodes = self.find_nodes_by_type(root_node, loop_type)
            for node in loop_nodes:
                if node is None:
                    continue
                loop_text = self.get_node_text(node, source_bytes)
                rules.append({
                    "type": "loop",
                    "subtype": loop_type,
                    "code": loop_text,
                    "location": self.get_location(node),
                    "file_path": relative_path  # 使用相对路径
                })
        
        # 查找所有 switch 语句
        switch_nodes = self.find_nodes_by_type(root_node, "switch_statement")
        for node in switch_nodes:
            if node is None:
                continue
            switch_text = self.get_node_text(node, source_bytes)
            rules.append({
                "type": "switch",
                "code": switch_text,
                "location": self.get_location(node),
                "file_path": relative_path  # 使用相对路径
            })
        
        return rules
    
    def extract_class_architectures(self, root_node, source_bytes, relative_path):
        """提取类架构信息并添加相对路径"""
        classes = []
        
        # 查找所有类定义
        class_nodes = self.find_nodes_by_type(root_node, "class_specifier")
        for node in class_nodes:
            if node is None:
                continue
                
            # 检查是否是 ROOT 特有的类定义
            class_text = self.get_node_text(node, source_bytes)
            if "ClassDef" in class_text or "ClassImp" in class_text:
                # 特殊处理 ROOT 类
                class_info = self.process_root_class(node, source_bytes, relative_path)
                if class_info:
                    classes.append(class_info)
                    continue
            
            class_info = {
                "name": "",
                "is_root_class": False,
                "base_classes": [],
                "methods": [],
                "fields": [],
                "location": self.get_location(node),
                "file_path": relative_path  # 使用相对路径
            }
            
            # 提取类名
            for child in node.children:
                if child is None:
                    continue
                if child.type == "type_identifier":
                    class_info["name"] = self.get_node_text(child, source_bytes)
                    break
            
            # 提取基类
            base_clause_nodes = self.find_nodes_by_type(node, "base_class_clause")
            for base_node in base_clause_nodes:
                if base_node is None:
                    continue
                for base_child in base_node.children:
                    if base_child is None or base_child.type != "base_class_specifier":
                        continue
                    base_class = self.get_node_text(base_child, source_bytes)
                    class_info["base_classes"].append(base_class)
            
            # 提取方法
            method_nodes = self.find_nodes_by_type(node, "function_definition")
            for method_node in method_nodes:
                if method_node is None:
                    continue
                    
                method_info = {
                    "name": "",
                    "return_type": "",
                    "parameters": [],
                    "location": self.get_location(method_node),
                    "file_path": relative_path  # 使用相对路径
                }
                
                # 提取方法名
                declarator = self.find_child_by_type(method_node, "function_declarator")
                if declarator:
                    identifier = self.find_child_by_type(declarator, "identifier")
                    if identifier:
                        method_info["name"] = self.get_node_text(identifier, source_bytes)
                
                # 提取返回类型
                type_node = self.find_child_by_type(method_node, "primitive_type") or \
                            self.find_child_by_type(method_node, "type_identifier")
                if type_node:
                    method_info["return_type"] = self.get_node_text(type_node, source_bytes)
                
                # 提取参数
                if declarator:
                    param_list = self.find_child_by_type(declarator, "parameter_list")
                    if param_list:
                        for param in param_list.children:
                            if param is None or param.type != "parameter_declaration":
                                continue
                                
                            param_type = self.find_child_by_type(param, "primitive_type") or \
                                        self.find_child_by_type(param, "type_identifier")
                            param_name = self.find_child_by_type(param, "identifier")
                            
                            param_info = {
                                "type": self.get_node_text(param_type, source_bytes) if param_type else "",
                                "name": self.get_node_text(param_name, source_bytes) if param_name else ""
                            }
                            method_info["parameters"].append(param_info)
                
                class_info["methods"].append(method_info)
            
            # 提取字段
            field_nodes = self.find_nodes_by_type(node, "field_declaration")
            for field_node in field_nodes:
                if field_node is None:
                    continue
                    
                field_info = {
                    "type": "",
                    "name": "",
                    "location": self.get_location(field_node),
                    "file_path": relative_path  # 使用相对路径
                }
                
                type_node = self.find_child_by_type(field_node, "primitive_type") or \
                            self.find_child_by_type(field_node, "type_identifier")
                if type_node:
                    field_info["type"] = self.get_node_text(type_node, source_bytes)
                
                declarator = self.find_child_by_type(field_node, "declarator")
                if declarator:
                    identifier = self.find_child_by_type(declarator, "identifier")
                    if identifier:
                        field_info["name"] = self.get_node_text(identifier, source_bytes)
                
                class_info["fields"].append(field_info)
            
            classes.append(class_info)
        
        return classes
    
    def process_root_class(self, node, source_bytes, relative_path):
        """处理 ROOT 特有的类定义并添加相对路径"""
        class_info = {
            "name": "",
            "is_root_class": True,
            "base_classes": [],
            "methods": [],
            "fields": [],
            "location": self.get_location(node),
            "file_path": relative_path  # 使用相对路径
        }
        
        # 尝试从 ClassDef 宏中提取类名
        class_def_node = self.find_nodes_by_type(node, "preproc_call")
        for macro_node in class_def_node:
            if macro_node is None:
                continue
            macro_text = self.get_node_text(macro_node, source_bytes)
            if "ClassDef" in macro_text:
                # 提取类名参数
                args_node = self.find_child_by_type(macro_node, "preproc_arg")
                if args_node:
                    args_text = self.get_node_text(args_node, source_bytes)
                    # 假设第一个参数是类名
                    class_name = args_text.split(",")[0].strip()
                    class_info["name"] = class_name
                    break
        
        # 如果无法从宏中提取类名，尝试从类定义中提取
        if not class_info["name"]:
            for child in node.children:
                if child and child.type == "type_identifier":
                    class_info["极"] = self.get_node_text(child, source_bytes)
                    break
        
        return class_info
    
    def extract_physics_constants(self, root_node, source_bytes, relative_path):
        """提取物理常量并添加相对路径"""
        constants = []
        
        # 查找所有常量定义
        const_nodes = self.find_nodes_by_type(root_node, "const")
        for node in const_nodes:
            if node is None:
                continue
                
            # 查找父节点是否是变量声明
            parent = node.parent
            if parent and parent.type == "variable_declarator":
                # 查找完整的声明
                declaration = parent.parent
                if declaration and declaration.type == "variable_declaration":
                    # 提取类型和名称
                    type_node = self.find_child_by_type(declaration, "primitive_type") or \
                                self.find_child_by_type(declaration, "type_identifier")
                    name_node = self.find_child_by_type(parent, "identifier")
                    
                    if type_node and name_node:
                        const_type = self.get_node_text(type_node, source_bytes)
                        const_name = self.get_node_text(name_node, source_bytes)
                        
                        # 查找初始值
                        value_node = self.find_child_by_type(parent, "number_literal") or \
                                     self.find_child_by_type(parent, "string_literal")
                        const_value = self.get_node_text(value_node, source_bytes) if value_node else ""
                        
                        constants.append({
                            "name": const_name,
                            "type": const_type,
                            "value": const_value,
                            "location": self.get_location(declaration),
                            "file_path": relative_path  # 使用相对路径
                        })
        
        # 查找 constexpr 定义
        constexpr_nodes = self.find_nodes_by_type(root_node, "constexpr")
        for node in constexpr_nodes:
            if node is None:
                continue
                
            # 查找父节点是否是变量声明
            parent = node.parent
            if parent and parent.type == "variable_declarator":
                # 查找完整的声明
                declaration = parent.parent
                if declaration and declaration.type == "variable_declaration":
                    # 提取类型和名称
                    type_node = self.find_child_by_type(declaration, "primitive_type") or \
                                self.find_child_by_type(declaration, "type_identifier")
                    name_node = self.find_child_by_type(parent, "identifier")
                    
                    if type_node and name_node:
                        const_type = self.get_node_text(type_node, source_bytes)
                        const_name = self.get_node_text(name_node, source_bytes)
                        
                        # 查找初始值
                        value_node = self.find_child_by_type(parent, "number_literal") or \
                                     self.find_child_by_type(parent, "string_literal")
                        const_value = self.get_node_text(value_node, source_bytes) if value_node else ""
                        
                        constants.append({
                            "name": const_name,
                            "type": const_type,
                            "value": const_value,
                            "location": self.get_location(declaration),
                            "file_path": relative_path  # 使用相对路径
                        })
        
        return constants
    
    def extract_root_macros(self, root_node, source_bytes, relative_path):
        """提取 ROOT 特有宏并添加相对路径"""
        macros = []
        
        # 查找所有预处理宏
        macro_nodes = self.find_nodes_by_type(root_node, "preproc_def")
        for node in macro_nodes:
            if node is None:
                continue
                
            macro_text = self.get_node_text(node, source_bytes)
            
            # 检查是否是 ROOT 特有宏
            if self.is_root_macro(macro_text):
                # 提取宏信息
                macro_info = self.extract_macro_info(node, source_bytes)
                if macro_info:
                    macro_info["file_path"] = relative_path  # 添加相对路径
                    macros.append(macro_info)
        
        # 查找宏调用
        macro_call_nodes = self.find_nodes_by_type(root_node, "preproc_call")
        for node in macro_call_nodes:
            if node is None:
                continue
                
            macro_text = self.get_node_text(node, source_bytes)
            
            # 检查是否是 ROOT 特有宏
            if self.is_root_macro(macro_text):
                # 提取宏调用信息
                macro_info = self.extract_macro_call_info(node, source_bytes)
                if macro_info:
                    macro_info["file_path"] = relative_path  # 添加相对路径
                    macros.append(macro_info)
        
        return macros
    
    def is_root_macro(self, macro_text):
        """判断是否是 ROOT 特有宏"""
        root_macro_keywords = ["R__", "ClassDef", "ClassImp", "ROOT", "TRoot", "TROOT"]
        return any(keyword in macro_text for keyword in root_macro_keywords)
    
    def extract_macro_info(self, node, source_bytes):
        """提取宏定义信息"""
        # 提取宏名称
        name_node = self.find_child_by_type(node, "identifier")
        macro_name = self.get_node_text(name_node, source_bytes) if name_node else ""
        
        # 提取宏值
        value_node = None
        for child in node.children:
            if child and child.type == "preproc_arg":
                value_node = child
                break
        
        macro_value = self.get_node_text(value_node, source_bytes) if value_node else ""
        
        return {
            "name": macro_name,
            "value": macro_value,
            "code": self.get_node_text(node, source_bytes),
            "location": self.get_location(node),
            "type": "definition"
        }
    
    def extract_macro_call_info(self, node, source_bytes):
        """提取宏调用信息"""
        # 提取宏名称
        name_node = self.find_child_by_type(node, "identifier")
        macro_name = self.get_node_text(name_node, source_bytes) if name_node else ""
        
        # 提取宏参数
        args_node = self.find_child_by_type(node, "preproc_arg")
        macro_args = self.get_node_text(args_node, source_bytes) if args_node else ""
        
        return {
            "name": macro_name,
            "arguments": macro_args,
            "code": self.get_node_text(node, source_bytes),
            "location": self.get_location(node),
            "type": "call"
        }
    
    # ====================== 辅助方法 ======================
    
    def find_nodes_by_type(self, root_node, node_type):
        """查找特定类型的所有节点（增强空节点检查）"""
        nodes = []
        
        def traverse(node):
            if node is None:
                return
            if node.type == node_type:
                nodes.append(node)
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return nodes
    
    def find_child_by_type(self, node, child_type):
        """查找特定类型的直接子节点（增强空节点检查）"""
        if node is None:
            return None
        for child in node.children:
            if child.type == child_type:
                return child
        return None
    
    def get_node_text(self, node, source_bytes):
        """获取节点对应的源代码文本（增强空节点检查）"""
        if node is None:
            return ""
        return source_bytes[node.start_byte:node.end_byte].decode('utf-8', 'ignore')
    
    def get_location(self, node):
        """获取节点位置信息（增强空节点检查）"""
        if node is None:
            return {
                "start_line": 0,
                "start_column": 0,
                "end_line": 0,
                "end_column": 0
            }
        start_line, start_col = node.start_point
        end_line, end_col = node.end_point
        return {
            "start_line": start_line + 1,
            "start_column": start_col + 1,
            "end_line": end_line + 1,
            "end_column": end_col + 1
        }

def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description='ROOT 框架代码分析器（相对路径版本）')
    parser.add_argument('repo_path', type=str, help='ROOT 源代码路径')
    parser.add_argument('-o', '--output', type=str, default='root_analysis.json', 
                       help='输出文件路径 (默认: root_analysis.json)')
    parser.add_argument('-d', '--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.repo_path):
        print(f"错误: '{args.repo_path}' 不是一个有效的目录")
        return 1
    
    print(f"=== 开始分析 ROOT 仓库: {args.repo_path} ===")
    
    try:
        analyzer = ROOTCodeAnalyzer(debug=args.debug)
        if not analyzer.parser:
            print("错误: 无法初始化解析器")
            return 2
        
        analyzer.analyze_repository(args.repo_path, args.output)
        return 0
    except Exception as e:
        print(f"严重错误: 分析过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    main()