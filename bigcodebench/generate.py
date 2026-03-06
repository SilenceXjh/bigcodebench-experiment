import os
from utils import load_tokenizer_model, load_jsonl_data, model_generate, extract_python_code


data_path = "/data0/xjh/bigcodebench-experiment/data/bigcodebench.jsonl"
model_path = "/data1/model/qwen/Qwen/Qwen2.5-Coder-7B-Instruct/"
output_path = "/data0/xjh/bigcodebench-experiment/qwen7b_generations"

os.makedirs(output_path, exist_ok=True)

data = load_jsonl_data(data_path)
tokenizer, model = load_tokenizer_model(model_path)

for sample in data[16:]:
    instruct_prompt = sample["instruct_prompt"]
    task_id = sample["task_id"]
    task_id = task_id.split("/")[-1]
    generated_text = model_generate(instruct_prompt, model, tokenizer)
    code = extract_python_code(generated_text)
    with open(os.path.join(output_path, f"{task_id}.py"), "w") as f:
        f.write('"""\n')
        f.write('instruction:\n')
        f.write(instruct_prompt + "\n")
        f.write('"""\n')
        f.write(code)