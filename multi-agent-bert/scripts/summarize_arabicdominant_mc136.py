"""Aggregate the FINAL 136-step matched-compute ArabicDominant ablation.

Two fixed corpora (Original-180 seed-42 subset, ArabicDominant-180) x model seeds
42/43/44 = 6 runs, 136 optimizer steps each, primary_only on the unchanged Silver-1163.

Reports per-run and mean +/- sd for accuracy / macro-F1 / weighted-F1, paired
AD - Original differences, per-class F1, and two collapse diagnostics per run:
number of distinct predicted classes and the largest predicted-class share.

COLLAPSE CRITERION (stated, not implied): a run is 'collapsed' if it predicts fewer
than all 9 classes, OR a single predicted class covers >= 50% of the 1163 samples.
For reference the healthy Topic-540 runs predict 9 classes with a top share of ~32%.

INTERPRETATION RULE (applied mechanically, not by judgement):
  all three paired differences same direction AND no run collapsed
      -> directional evidence for/against Arabic-ratio alignment
  otherwise -> inconclusive
No significance is claimed from three seeds.

The 48-step results live in a different directory and are NEVER pooled with these.
"""
import json, pathlib, statistics
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTR = ROOT / "experiments/outputs/multi_agent_bert/experiment_topic_arabicdominant_mc136"
SEEDS = [42, 43, 44]
ARMS = {"orig": "Original-180", "ad": "ArabicDominant-180"}
LABELS = ["business", "education", "health", "shopping", "medical",
          "sports", "tech", "finance", "social"]
COLLAPSE_SHARE = 0.50


def analyse(rows):
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
    pred = Counter(r["predicted_label"] for r in rows)
    top_lab, top_n = pred.most_common(1)[0]
    n_classes = len(pred)
    top_share = top_n / n
    return {
        "n": n, "acc": acc,
        "macro_f1": sum(f1s.values()) / len(LABELS),
        "weighted_f1": sum(f1s[l] * support[l] for l in LABELS) / n,
        "per_class_f1": f1s,
        "n_predicted_classes": n_classes,
        "top_pred_label": top_lab, "top_pred_share": top_share,
        "collapsed": (n_classes < len(LABELS)) or (top_share >= COLLAPSE_SHARE),
    }


res = {}
for arm in ARMS:
    for seed in SEEDS:
        d = OUTR / f"{arm}180_mseed{seed}"
        pj = list(d.glob("*_predictions.json"))
        if not pj:
            raise SystemExit(f"missing predictions: {d}")
        rows = json.load(open(pj[0], encoding="utf-8"))
        res[(arm, seed)] = analyse(rows)

lines = []


def out(s=""):
    print(s)
    lines.append(s)


out("# ArabicDominant-180 vs Original-180 - matched-compute ablation at 136 steps")
out()
out("Two FIXED balanced corpora, model seeds 42/43/44, exactly 136 optimizer steps each.")
out("Identical preprocessing, hyperparameters, label mapping, validation procedure (no dev),")
out("checkpoint selection (final step, load_best off) and Silver-1163 primary_only evaluation.")
out("The training corpus is the only difference. No data was generated for this run.")
out()
out(f"Collapse criterion: fewer than 9 predicted classes, or a top predicted-class share >= "
    f"{COLLAPSE_SHARE:.0%}. (Healthy Topic-540 reference: 9 classes, top share ~32%.)")
out()
out("## Per-run, Silver-1163")
out()
out("| arm | seed | acc | macro-F1 | weighted-F1 | #pred classes | top class | top share | collapsed |")
out("|---|--:|--:|--:|--:|--:|---|--:|---|")
for arm, name in ARMS.items():
    for seed in SEEDS:
        m = res[(arm, seed)]
        out(f"| {name} | {seed} | {m['acc']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} "
            f"| {m['n_predicted_classes']} | {m['top_pred_label']} | {m['top_pred_share']:.1%} "
            f"| {'YES' if m['collapsed'] else 'no'} |")

out()
out("## Mean +/- sd over the three model seeds")
out()
out("| arm | acc | macro-F1 | weighted-F1 |")
out("|---|---|---|---|")
for arm, name in ARMS.items():
    cols = []
    for k in ("acc", "macro_f1", "weighted_f1"):
        xs = [res[(arm, s)][k] for s in SEEDS]
        cols.append(f"{statistics.mean(xs):.4f} +/- {statistics.pstdev(xs):.4f}")
    out(f"| {name} | {cols[0]} | {cols[1]} | {cols[2]} |")

out()
out("## Paired differences (ArabicDominant - Original), same model seed")
out()
out("| seed | d acc | d macro-F1 | d weighted-F1 |")
out("|--:|--:|--:|--:|")
d = {k: [] for k in ("acc", "macro_f1", "weighted_f1")}
for seed in SEEDS:
    a, o = res[("ad", seed)], res[("orig", seed)]
    for k in d:
        d[k].append(a[k] - o[k])
    out(f"| {seed} | {a['acc']-o['acc']:+.4f} | {a['macro_f1']-o['macro_f1']:+.4f} "
        f"| {a['weighted_f1']-o['weighted_f1']:+.4f} |")
out(f"| **mean** | **{statistics.mean(d['acc']):+.4f}** | **{statistics.mean(d['macro_f1']):+.4f}** "
    f"| **{statistics.mean(d['weighted_f1']):+.4f}** |")

out()
out("## Per-class F1 (mean over seeds)")
out()
out("| class | Original-180 | ArabicDominant-180 | delta |")
out("|---|--:|--:|--:|")
for lab in LABELS:
    o = statistics.mean(res[("orig", s)]["per_class_f1"][lab] for s in SEEDS)
    a = statistics.mean(res[("ad", s)]["per_class_f1"][lab] for s in SEEDS)
    out(f"| {lab} | {o:.3f} | {a:.3f} | {a-o:+.3f} |")

# ---- mechanical interpretation ----
any_collapsed = [f"{arm}/{s}" for arm in ARMS for s in SEEDS if res[(arm, s)]["collapsed"]]
consistent = {k: len({x > 0 for x in d[k]}) == 1 for k in d}
out()
out("## Interpretation (rule applied mechanically)")
out()
out(f"- collapsed runs: {any_collapsed if any_collapsed else 'NONE'}")
for k in ("acc", "macro_f1", "weighted_f1"):
    signs = " ".join(f"{x:+.4f}" for x in d[k])
    out(f"- {k}: {signs} -> {'consistent' if consistent[k] else 'MIXED'}")
verdict_ok = (not any_collapsed) and consistent["acc"] and consistent["macro_f1"]
if verdict_ok:
    direction = "FOR" if statistics.mean(d["acc"]) > 0 else "AGAINST"
    out()
    out(f"**VERDICT: directional evidence {direction} Arabic-ratio alignment.** All three paired "
        f"differences share a direction on both accuracy and macro-F1, and no run collapsed. "
        f"Three seeds; no significance claimed.")
else:
    reason = []
    if any_collapsed:
        reason.append("at least one run remains collapsed")
    if not (consistent["acc"] and consistent["macro_f1"]):
        reason.append("paired difference signs are mixed")
    out()
    out(f"**VERDICT: INCONCLUSIVE** ({'; '.join(reason)}). No directional claim is made.")
out()
out("These 136-step results are NOT combined with the 48-step diagnostic, which is an")
out("undertraining artefact and does not demonstrate a cs_ratio effect.")

OUTR.mkdir(parents=True, exist_ok=True)
(OUTR / "MC136_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
json.dump({f"{a}_{s}": res[(a, s)] for a in ARMS for s in SEEDS},
          open(OUTR / "mc136_summary.json", "w", encoding="utf-8"), indent=2)
print(f"\nwrote {OUTR / 'MC136_RESULTS.md'}")
