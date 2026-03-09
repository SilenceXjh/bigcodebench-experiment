import os
from utils import load_tokenizer_model, load_jsonl_data, model_generate, extract_python_code
import ast


data_path = "/data0/xjh/bigcodebench-experiment/data/bigcodebench.jsonl"
model_path = "/data1/model/qwen/Qwen/Qwen2.5-Coder-7B-Instruct/"
output_path = "/data0/xjh/bigcodebench-experiment/qwen7b_test_first_generations"

os.makedirs(output_path, exist_ok=True)


def trim_test_class(source: str) -> str:
    """
    保留 setUp/tearDown 方法和第一个 test 方法，删除其余 test 方法和 import 语句。
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    lines_to_remove = set()

    for node in ast.walk(tree):
        # 删除 import 语句
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for i in range(node.lineno - 1, node.end_lineno):
                lines_to_remove.add(i)

        # 处理类中的 test 方法
        if not isinstance(node, ast.ClassDef):
            continue

        first_test_seen = False
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            if item.name.startswith('test'):
                if not first_test_seen:
                    first_test_seen = True
                else:
                    for i in range(item.lineno - 1, item.end_lineno):
                        lines_to_remove.add(i)

    result_lines = [
        line for i, line in enumerate(lines)
        if i not in lines_to_remove
    ]

    return ''.join(result_lines)

data = load_jsonl_data(data_path)
tokenizer, model = load_tokenizer_model(model_path)

for sample in data:
    comlete_prompt = sample["complete_prompt"]
    task_id = sample["task_id"]
    task_id = task_id.split("/")[-1]
    test_code = sample["test"]
    # print("-----original test-----")
    # print(test_code)
    trimed_test_code = trim_test_class(test_code)
    # print("-----trimed test code-----")
    # print(trimed_test_code)

    gen_test_prompt = f"""Generate some testcases for a function. The function head is as follows:
```
{comlete_prompt}
```
You can generate testcase like this:
```
{trimed_test_code}
```
Generated 3-5 testcases in a 'unittest.TestCase' class. Only output the generated testcases without any explanation.
"""
    # print("-----gen test prompt-----")
    # print(gen_test_prompt)
    generated_text = model_generate(gen_test_prompt, model, tokenizer)
    # print("-----gen text-----")
    # print(generated_text)
    gen_test = extract_python_code(generated_text)

    prompt = sample["instruct_prompt"]
    prompt += "\nYour implementation should pass the following test:\n"
    prompt += f"```\n{gen_test}\n```\n"
    prompt += "Only output the function code you implemented."

    # print(prompt)

    generated_text = model_generate(prompt, model, tokenizer)
    code = extract_python_code(generated_text)
    with open(os.path.join(output_path, f"{task_id}.py"), "w") as f:
        f.write('"""\n')
        f.write('instruction:\n')
        f.write(sample["instruct_prompt"] + "\n")
        f.write('"""\n')
        f.write(code)
