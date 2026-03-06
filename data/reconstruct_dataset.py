import json

target_file = "bigcodebench.json"
data = []

with open("bigcodebench.jsonl", "r") as f:
    for line in f.readlines():
        sample = json.loads(line)
        data.append(sample)

with open(target_file, "w") as f:
    json.dump(data, f, indent=2)