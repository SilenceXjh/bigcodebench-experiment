import os
from utils import load_tokenizer_model, load_jsonl_data, model_generate, extract_python_code
from evaluate import run_single_sample


data_path = "/data0/xjh/bigcodebench-experiment/data/bigcodebench.jsonl"
model_path = "/data1/model/qwen/Qwen/Qwen2.5-Coder-1.5B-Instruct/"
code_path = "/data0/xjh/bigcodebench-experiment/qwen1.5b_test_first_generations"
output_path = "/data0/xjh/bigcodebench-experiment/qwen1.5b_repairs"

os.makedirs(output_path, exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

data = load_jsonl_data(data_path)
tokenizer, model = load_tokenizer_model(model_path)


def construct_repair_prompt(instruct_prompt, code, test, feedback):
    prompt = f"""Please fix a python function base on the function description, current code, test code, and test feedback.

### Function Description:
{instruct_prompt}
        
### Current code:
```python
{code}
```

### Test code:
```
{test}
```

### Test feedBack: 
{feedback}

Provide only the fixed Python function code without any explanation."""
    return prompt


total = 0
right = 0
repair_num = [0,0,0]

for sample in data:
    total += 1

    instruct_prompt = sample["instruct_prompt"]
    task_id = sample["task_id"]
    task_id = task_id.split("/")[-1]

    with open(os.path.join(code_path, f"{task_id}.py"), "r") as f:
        code = f.read()
        prefix = '"""\n' + 'instruction:\n' + instruct_prompt + "\n" + '"""\n'
        prefix_len = len(prefix)
        code = code[prefix_len:]

    test = sample["test"]
    success, feedback = run_single_sample(code, test)

    if success:
        right += 1
    else:
        for i in range(3):
            prompt = construct_repair_prompt(instruct_prompt, code, test, feedback)
            # print("-----prompt-----")
            # print(prompt)
            generated_text = model_generate(prompt, model, tokenizer)
            code = extract_python_code(generated_text)
            success, feedback = run_single_sample(code, test)
            if success:
                repair_num[i] += 1
                with open(os.path.join(output_path, f"{task_id}.py"), "w") as f:
                    f.write(code)
                break

print("total:", total)
print("right:", right)
print("repaired:", repair_num)