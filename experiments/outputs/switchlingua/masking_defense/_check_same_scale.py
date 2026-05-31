"""Proof that the aggregate score and the per-sentence scores use the SAME
formula — so comparing them isolates ONLY the decision rule (average vs each).
Reads existing System B output. No API calls."""
import json, pathlib, statistics

ROOT = pathlib.Path(__file__).resolve().parents[4]
B = ROOT / "experiments" / "outputs" / "switchlingua" / "system_b_modified_mini" / "Arabic.jsonl"

print("Checking: is the aggregate score = the average of the per-sentence scores?\n")
print(f"{'rec':>3} | {'aggregate (score)':>17} | {'avg of sentences':>17} | {'match?':>6}")
print("-" * 56)
with open(B, encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        agg = r.get("score")
        sent = r.get("sentence_scores", [])
        if not sent or agg is None:
            print(f"{i:>3} | (no sentence_scores)")
            continue
        avg = statistics.mean(sent)
        match = "YES" if abs(float(agg) - avg) < 0.01 else "no"
        print(f"{i:>3} | {float(agg):>17.4f} | {avg:>17.4f} | {match:>6}")
