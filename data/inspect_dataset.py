import json

with open("bigcodebench.json", "r") as f:
     data = json.load(f)
     print("dataset size:", len(data))
     for sample in data[:3]:
          print(sample["task_id"])
          print("doc struct")
          print(sample["doc_struct"])
          print("instruct prompt")
          print(sample["instruct_prompt"])
          print("test")
          print(sample["test"])