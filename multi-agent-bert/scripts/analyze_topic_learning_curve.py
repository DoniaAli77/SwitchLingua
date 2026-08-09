"""Analyse the SwitchLingua topic learning curve (180/360/540 x seeds 42/43/44)
evaluated on the unchanged Silver-1163 corpus. Read-only over saved predictions.
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).parent.parent
D = ROOT / "experiments/outputs/multi_agent_bert/experiment_topic_learning_curve"
from sklearn.metrics import accuracy_score, f1_score

LABELS = ["business", "education", "health", "shopping", "medical",
          "sports", "tech", "finance", "social"]
SIZES = [180, 360, 540]
SEEDS = [42, 43, 44]

res = {}
for size in SIZES:
    for seed in SEEDS:
        tag = f"topic{size}_seed{seed}"
        p = D / tag / f"{tag}__full_pipeline_predictions.csv"
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        assert len(rows) == 1163, (tag, len(rows))
        yt = [r["true_label"] for r in rows]
        yp = [r["predicted_label"] for r in rows]
        res[(size, seed)] = {
            "correct": sum(a == b for a, b in zip(yt, yp)),
            "accuracy": accuracy_score(yt, yp),
            "macro_f1": f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0),
            "weighted_f1": f1_score(yt, yp, labels=LABELS, average="weighted", zero_division=0),
        }

print("=" * 78)
print("PER-RUN RESULTS on Silver-1163 (n=1163, primary_only)")
print("=" * 78)
print(f"{'size':>5} {'seed':>5} {'correct':>8} {'accuracy':>10} {'macro_F1':>10} {'weighted_F1':>12}")
for size in SIZES:
    for seed in SEEDS:
        r = res[(size, seed)]
        print(f"{size:>5} {seed:>5} {r['correct']:>8} {r['accuracy']:>10.4f} "
              f"{r['macro_f1']:>10.4f} {r['weighted_f1']:>12.4f}")

print("\n" + "=" * 78)
print("MEAN +/- SD BY TRAINING SIZE (sample SD, n=3)")
print("=" * 78)
agg = {}
print(f"{'size':>5} | {'accuracy':>18} | {'macro_F1':>18} | {'weighted_F1':>18}")
for size in SIZES:
    a = [res[(size, s)]["accuracy"] for s in SEEDS]
    m = [res[(size, s)]["macro_f1"] for s in SEEDS]
    w = [res[(size, s)]["weighted_f1"] for s in SEEDS]
    agg[size] = {"accuracy": (st.mean(a), st.stdev(a)),
                 "macro_f1": (st.mean(m), st.stdev(m)),
                 "weighted_f1": (st.mean(w), st.stdev(w))}
    print(f"{size:>5} | {st.mean(a):.4f} +/- {st.stdev(a):.4f} | "
          f"{st.mean(m):.4f} +/- {st.stdev(m):.4f} | {st.mean(w):.4f} +/- {st.stdev(w):.4f}")

print("\n" + "=" * 78)
print("PAIRED MACRO-F1 CHANGES (same seed, nested data)")
print("=" * 78)
pairs = {}
for lo, hi in [(180, 360), (360, 540)]:
    d = [res[(hi, s)]["macro_f1"] - res[(lo, s)]["macro_f1"] for s in SEEDS]
    pairs[f"{lo}->{hi}"] = d
    print(f"\n{lo} -> {hi}:")
    for s, v in zip(SEEDS, d):
        print(f"   seed {s}: {res[(lo,s)]['macro_f1']:.4f} -> {res[(hi,s)]['macro_f1']:.4f}   delta = {v:+.4f}")
    print(f"   mean delta = {st.mean(d):+.4f} +/- {st.stdev(d):.4f}   "
          f"(all seeds same sign: {all(x>0 for x in d) or all(x<0 for x in d)})")

print("\n" + "=" * 78)
print("REPRODUCTION CHECK: 540 / seed 42  vs  existing primary result")
print("=" * 78)
r = res[(540, 42)]
REF_ACC, REF_MF1 = 0.6242, 0.5599
print(f"existing primary : accuracy {REF_ACC:.4f}   macro_F1 {REF_MF1:.4f}")
print(f"this run (540/42): accuracy {r['accuracy']:.4f}   macro_F1 {r['macro_f1']:.4f}")
print(f"difference       : accuracy {r['accuracy']-REF_ACC:+.4f}   macro_F1 {r['macro_f1']-REF_MF1:+.4f}")
# compare against the seed spread at 540
sd_a, sd_m = agg[540]["accuracy"][1], agg[540]["macro_f1"][1]
print(f"540 seed-spread SD: accuracy {sd_a:.4f}   macro_F1 {sd_m:.4f}")
print(f"|diff| within 1 SD of seed spread? accuracy={abs(r['accuracy']-REF_ACC)<=sd_a}  "
      f"macro_F1={abs(r['macro_f1']-REF_MF1)<=sd_m}")
# exact prediction-level comparison with the stored baseline
base = list(csv.DictReader(open(ROOT / "experiments/outputs/multi_agent_bert/experiment_silver_topic540/xlmr_combined_1163_predictions.csv", encoding="utf-8")))
new = list(csv.DictReader(open(D / "topic540_seed42/topic540_seed42__full_pipeline_predictions.csv", encoding="utf-8")))
same = sum(1 for a, b in zip(base, new) if a["predicted_label"] == b["predicted_label"])
print(f"identical predictions vs stored baseline: {same}/1163 ({same/1163*100:.2f}%)")

out = {
    "per_run": {f"{k[0]}_seed{k[1]}": {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                                        for kk, vv in v.items()} for k, v in res.items()},
    "by_size_mean_sd": {str(s): {k: {"mean": round(v[0], 6), "sd": round(v[1], 6)}
                                  for k, v in agg[s].items()} for s in SIZES},
    "paired_macro_f1_deltas": {k: [round(x, 6) for x in v] for k, v in pairs.items()},
    "reproduction_check_540_seed42": {
        "reference_accuracy": REF_ACC, "reference_macro_f1": REF_MF1,
        "observed_accuracy": round(res[(540, 42)]["accuracy"], 6),
        "observed_macro_f1": round(res[(540, 42)]["macro_f1"], 6),
        "identical_predictions_vs_baseline": same, "n": 1163,
    },
}
json.dump(out, open(D / "learning_curve_summary.json", "w", encoding="utf-8"), indent=2)
print(f"\nwrote -> {D/'learning_curve_summary.json'}")
