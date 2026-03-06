import ast
import os
import sys
import json


def extract_imports(code: str):
    """
    解析 import 语句
    """
    try:
        tree = ast.parse(code)
    except Exception as e:
        print(e)
        print(code)
        return []
    modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                modules.add(n.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split('.')[0])

    std_lib = sys.stdlib_module_names
    modules = list(modules)
    third_party = [m for m in modules if m not in std_lib]

    return third_party


def main():
    code_dir = "/data0/xjh/bigcodebench-experiment/qwen7b_generations"
    third_party_set = set()

    for file_name in os.listdir(code_dir):
        with open(os.path.join(code_dir, file_name), "r") as f:
            code = f.read()
        third_party = extract_imports(code)
        third_party_set.update(third_party)

    print(list(third_party_set))

main()