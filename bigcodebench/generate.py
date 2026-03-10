import os

from openai import OpenAI
from utils import load_tokenizer_model, load_jsonl_data, model_generate, extract_python_code, ds_api_generate


data_path = "/data0/xjh/bigcodebench-experiment/data/bigcodebench.jsonl"
model_path = "/data1/model/qwen/Qwen/Qwen2.5-Coder-1.5B-Instruct/"
output_path = "/data0/xjh/bigcodebench-experiment/ds_generations"

USE_DS_API = True

os.makedirs(output_path, exist_ok=True)

data = load_jsonl_data(data_path)

if USE_DS_API:
    client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
else:
    tokenizer, model = load_tokenizer_model(model_path)

for sample in data[:1]:
    instruct_prompt = sample["instruct_prompt"]
    task_id = sample["task_id"]
    task_id = task_id.split("/")[-1]
    if USE_DS_API:
        generated_text = ds_api_generate(instruct_prompt, client)
    else:
        generated_text = model_generate(instruct_prompt, model, tokenizer)
    code = extract_python_code(generated_text)
    with open(os.path.join(output_path, f"{task_id}.py"), "w") as f:
        f.write('"""\n')
        f.write('instruction:\n')
        f.write(instruct_prompt + "\n")
        f.write('"""\n')
        f.write(code)