from datasets import load_dataset

dataset = load_dataset("bigcode/bigcodebench", split="v0.1.4")

dataset.to_json("bigcodebench.jsonl")