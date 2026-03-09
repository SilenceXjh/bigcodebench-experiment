import json

with open("bigcodebench.json", "r") as f:
     data = json.load(f)
     print("dataset size:", len(data))
     for sample in data[:3]:
          print(sample["task_id"])
          print("complete_prompt")
          print(sample["complete_prompt"])
          print("test")
          print(sample["test"])