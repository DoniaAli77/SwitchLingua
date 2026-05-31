"""Two questions, answered from real System B output. No API calls.
  Q1: Does task-validation enter the sentence SCORE? (should be NO)
  Q2: Did the refiner already run on the saved data? (matters for masking)
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[4]
B = ROOT / "experiments" / "outputs" / "switchlingua" / "system_b_modified_mini" / "Arabic.jsonl"

print("The score formula (from utils.py) is:")
print("   score = fluency*0.3 + naturalness*0.25 + cs_ratio*0.2 + socio*0.25")
print("   --> NO task-validation term. Task validation is a separate yes/no flag.\n")

print(f"{'rec':>3} | {'refine happened?':>16} | {'instance_refine_counts'}")
print("-" * 60)
with open(B, encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rc = r.get("refine_count", 0)
        irc = r.get("instance_refine_counts", [])
        happened = "YES" if (rc and int(rc) > 0) or any(irc) else "no"
        print(f"{i:>3} | {happened:>16} | {irc}")
