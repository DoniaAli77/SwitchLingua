import json
import os
from collections import Counter

from utils import load_config, generate_scenarios

base_dir = os.path.dirname(__file__)
config_path = os.path.join(base_dir, "..", "config", "config2.yaml")
config = load_config(config_path)

scenarios = generate_scenarios(config["pre_execute"])

counts = Counter(s["task"] for s in scenarios)
print("total:", len(scenarios))
print("by_task:", counts)

output_dir = os.path.join(base_dir, "..", "output")
os.makedirs(output_dir, exist_ok=True)
jsonl_path = os.path.join(output_dir, "task_aware_scenarios.jsonl")
summary_path = os.path.join(output_dir, "task_aware_scenarios_summary.txt")

with open(jsonl_path, "w", encoding="utf-8") as f:
    for scenario in scenarios:
        f.write(json.dumps(scenario, ensure_ascii=True))
        f.write("\n")

def validate(s):
    task = s["task"]
    if task == "topic":
        assert "label" in s
    elif task == "sentiment":
        assert "label" in s and "intensity" in s["task_constraints"]
    elif task == "ner":
        assert "annotations" in s and "entity_types" in s["task_constraints"]

for task in ["topic", "sentiment", "ner"]:
    sample = next(s for s in scenarios if s["task"] == task)
    validate(sample)
    print(f"sample[{task}]:", sample)

with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"total: {len(scenarios)}\n")
    f.write(f"by_task: {dict(counts)}\n")
    for task in ["topic", "sentiment", "ner"]:
        sample = next(s for s in scenarios if s["task"] == task)
        f.write(f"sample[{task}]: {sample}\n")

print("wrote:", jsonl_path)
print("wrote:", summary_path)