import json
import sys
from pathlib import Path
from statistics import mean, stdev

# Load records
output_path = "../../output/Arabic.jsonl"
records = []

with open(output_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

print("="*80)
print("PIPELINE TEST RESULTS")
print("="*80)
print(f"\nTotal Records Processed: {len(records)}\n")

# Metrics
scores = []
passed_count = 0
fluency_all = []
cs_ratio_scores = []

for record in records:
    scores.append(record.get('score', 0))
    
    if record.get('task_validation_result', {}).get('passed'):
        passed_count += 1
    
    fluency = record.get('fluency_results_per_instances', [])
    if fluency:
        fluency_all.extend(fluency)
    
    cs_results = record.get('cs_ratio_results_per_instances', [])
    for cs in cs_results:
        if isinstance(cs, dict):
            cs_ratio_scores.append(cs.get('ratio_score', 0))

# Print summary
print("SUMMARY METRICS")
print("-" * 80)
print(f"Total Records: {len(records)}")
print(f"Task Validation Pass Rate: {(passed_count/len(records)*100):.1f}% ({passed_count}/{len(records)})")
print()

print("Score Distribution:")
print(f"  Average Score: {mean(scores):.2f}")
print(f"  Min Score: {min(scores):.2f}")
print(f"  Max Score: {max(scores):.2f}")
print(f"  Std Dev: {stdev(scores) if len(scores) > 1 else 0:.2f}")
print()

print("Per-Instance Evaluation:")
print(f"  Average Fluency: {mean(fluency_all):.2f}")
print(f"  Average CS Ratio Score: {mean(cs_ratio_scores):.2f}")
print()

# Task breakdown
tasks = {}
for record in records:
    task = record.get('task', 'unknown')
    if task not in tasks:
        tasks[task] = {'count': 0, 'passed': 0, 'scores': []}
    tasks[task]['count'] += 1
    tasks[task]['scores'].append(record.get('score', 0))
    if record.get('task_validation_result', {}).get('passed'):
        tasks[task]['passed'] += 1

print("\nPERFORMANCE BY TASK")
print("-" * 80)
for task, data in sorted(tasks.items()):
    avg = mean(data['scores']) if data['scores'] else 0
    pass_rate = (data['passed'] / data['count'] * 100) if data['count'] > 0 else 0
    print(f"{task.upper()}:")
    print(f"  Count: {data['count']}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"  Avg Score: {avg:.2f}")

# Sample records
print("\n\n" + "="*80)
print("SAMPLE RECORDS (First 3)")
print("="*80)

for idx, record in enumerate(records[:3], 1):
    print(f"\nRECORD {idx}")
    print("-" * 80)
    print(f"Task: {record.get('task')} | Label: {record.get('label')}")
    print(f"Validation: {'PASSED' if record.get('task_validation_result', {}).get('passed') else 'FAILED'}")
    
    data_gen = record.get('data_generation_result', [])
    print(f"\nGenerated ({len(data_gen)} sentences):")
    for i, sent in enumerate(data_gen[:3], 1):
        print(f"  [{i}] {sent[:100]}...")
    
    fluency = record.get('fluency_results_per_instances', [])
    print(f"\nScores:")
    print(f"  Fluency: {fluency}")
    print(f"  Final Score: {record.get('score'):.2f}")

print("\n" + "="*80)
print("QUALITY VERDICT")
print("="*80)

verdicts = []
if mean(scores) >= 8.0:
    verdicts.append("[OK] Average score >= 8.0")
else:
    verdicts.append(f"[NEED WORK] Average score {mean(scores):.2f} < 8.0 target")

if passed_count/len(records)*100 >= 80:
    verdicts.append("[OK] Validation pass rate >= 80%")
else:
    verdicts.append(f"[NEED WORK] Validation pass rate {passed_count/len(records)*100:.1f}% < 80% target")

for v in verdicts:
    print(f"  {v}")

print("\n" + "="*80)
