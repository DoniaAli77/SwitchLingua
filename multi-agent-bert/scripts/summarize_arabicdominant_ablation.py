"""Aggregate the ArabicDominant-180 vs Original-180 configuration ablation.

Reads the six saved per-sample prediction files (2 arms x seeds 42/43/44), recomputes
accuracy / macro-F1 / weighted-F1 from scratch, and reports the arms separately plus the
per-seed paired deltas. Also reports metrics on the Silver-1044 primary subset, which is
rows 0-1043 of the ordered Silver-1163 file (positional sample_ids).

Exploratory: 3 seeds, single dataset per arm. No significance claim is made.
Read-only apart from the two summary files it writes into the ablation output dir.
"""
import json, pathlib, statistics
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTR = ROOT / "experiments/outputs/multi_agent_bert/experiment_topic_arabicdominant_ablation"
N_PRIMARY = 1044
SEEDS = [42, 43, 44]
ARMS = {"orig": "Original-180", "ad": "ArabicDominant-180"}
LABELS = ["business", "education", "health", "shopping", "medical",
          "sports", "tech", "finance", "social"]


def prf(rows):
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    f1s, support = {}, Counter(r["true_label"] for r in rows)
    for lab in LABELS:
        tp = sum(1 for r in rows if r["true_label"] == lab and r["predicted_label"] == lab)
        fp = sum(1 for r in rows if r["true_label"] != lab and r["predicted_label"] == lab)
        fn = sum(1 for r in rows if r["true_label"] == lab and r["predicted_label"] != lab)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s[lab] = 2 * p * rc / (p + rc) if p + rc else 0.0
    macro = sum(f1s.values()) / len(LABELS)
    weighted = sum(f1s[l] * support[l] for l in LABELS) / n
    return {"n": n, "acc": acc, "macro_f1": macro, "weighted_f1": weighted, "per_class_f1": f1s}


def load(arm, seed):
    d = OUTR / f"{arm}180_seed{seed}"
    pj = list(d.glob("*_predictions.json"))
    if not pj:
        raise SystemExit(f"missing predictions for {arm} seed {seed} ({d})")
    rows = json.load(open(pj[0], encoding="utf-8"))
    assert rows[0]["sample_id"] == "sample_00000", "prediction order is not positional"
    return rows


res = {}
for arm in ARMS:
    for seed in SEEDS:
        rows = load(arm, seed)
        res[(arm, seed)] = {"all": prf(rows), "p1044": prf(rows[:N_PRIMARY])}

lines = []


def out(s=""):
    print(s)
    lines.append(s)


out("# ArabicDominant-180 vs Original-180 — exploratory configuration ablation")
out()
out("Single factor: generation cs_ratio [50%,60%] -> [85%,95%]. Identical training recipe")
out("(xlm-roberta-base, 4 epochs = 48 steps, batch 16, lr 2e-5, maxlen 256, fp16, adamw_torch,")
out("load_best off, no dev), identical preprocessing, seeds 42/43/44, primary_only evaluation")
out("on the unchanged Silver-1163. Exploratory: 3 seeds, no significance test.")
out()
out("## Per-run, Silver-1163")
out()
out("| arm | seed | acc | macro-F1 | weighted-F1 |")
out("|---|--:|--:|--:|--:|")
for arm, name in ARMS.items():
    for seed in SEEDS:
        m = res[(arm, seed)]["all"]
        out(f"| {name} | {seed} | {m['acc']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} |")

out()
out("## Arms reported separately")
out()
out("| arm | set | acc | macro-F1 | weighted-F1 |")
out("|---|---|---|---|---|")
agg = {}
for arm, name in ARMS.items():
    for st, lab in (("all", "Silver-1163"), ("p1044", "Silver-1044")):
        cols = []
        for k in ("acc", "macro_f1", "weighted_f1"):
            xs = [res[(arm, s)][st][k] for s in SEEDS]
            agg[(arm, st, k)] = (statistics.mean(xs), statistics.pstdev(xs))
            cols.append(f"{statistics.mean(xs):.4f} ± {statistics.pstdev(xs):.4f}")
        out(f"| {name} | {lab} | {cols[0]} | {cols[1]} | {cols[2]} |")

out()
out("## Paired per-seed deltas (ArabicDominant - Original), Silver-1163")
out()
out("| seed | Δ acc | Δ macro-F1 | Δ weighted-F1 |")
out("|--:|--:|--:|--:|")
for seed in SEEDS:
    a, o = res[("ad", seed)]["all"], res[("orig", seed)]["all"]
    out(f"| {seed} | {a['acc']-o['acc']:+.4f} | {a['macro_f1']-o['macro_f1']:+.4f} | "
        f"{a['weighted_f1']-o['weighted_f1']:+.4f} |")
da = [res[("ad", s)]["all"]["acc"] - res[("orig", s)]["all"]["acc"] for s in SEEDS]
dm = [res[("ad", s)]["all"]["macro_f1"] - res[("orig", s)]["all"]["macro_f1"] for s in SEEDS]
out(f"| **mean** | **{statistics.mean(da):+.4f}** | **{statistics.mean(dm):+.4f}** | |")
out()
out(f"Direction consistent across all three seeds: "
    f"{'YES' if len({d > 0 for d in da}) == 1 else 'NO'} (accuracy), "
    f"{'YES' if len({d > 0 for d in dm}) == 1 else 'NO'} (macro-F1).")

out()
out("## Per-class F1, mean over seeds, Silver-1163")
out()
out("| class | Original-180 | ArabicDominant-180 | Δ |")
out("|---|--:|--:|--:|")
for lab in LABELS:
    o = statistics.mean(res[("orig", s)]["all"]["per_class_f1"][lab] for s in SEEDS)
    a = statistics.mean(res[("ad", s)]["all"]["per_class_f1"][lab] for s in SEEDS)
    out(f"| {lab} | {o:.3f} | {a:.3f} | {a-o:+.3f} |")

out()
out("## Caveats")
out()
out("- The Original arm uses a different nested 180-subset per seed, so its spread includes")
out("  subset-sampling variance; the ArabicDominant arm has one fixed 180 corpus (205 accepted")
out("  sentences total), so its spread is training variance only.")
out("- 3 seeds, one dataset per arm: exploratory, no significance test.")
out("- Silver is final evaluation only; it never informed generation, training or selection.")

OUTR.mkdir(parents=True, exist_ok=True)
(OUTR / "ABLATION_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
json.dump({f"{a}_{s}": {k: v for k, v in res[(a, s)].items()} for a in ARMS for s in SEEDS},
          open(OUTR / "ablation_summary.json", "w", encoding="utf-8"), indent=2)
print(f"\nwrote {OUTR/'ABLATION_RESULTS.md'}")
