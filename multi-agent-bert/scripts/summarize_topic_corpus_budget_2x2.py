"""Corpus x budget 2x2 for the generated topic corpora, evaluated on Silver-1163.

    corpus \\ budget      136 steps            272 steps
    Topic-540            mc540_seed*          mc540x2_seed*
    Topic-1080           mc1080_seed*         mc1080x2_seed*

All eight... (six existing + the new three) runs share one recipe - xlm-roberta-base,
batch 16, grad_accum 1, lr 2e-5, maxlen 256, fp16, adamw_torch, no dev, load_best off,
9 labels in the frozen order - and primary_only evaluation on the unchanged Silver-1163.
Only the training corpus and --max_steps differ.

Three paired contrasts are reported separately because they answer different questions:
  A. EPOCH-MATCHED  1080@272 (4.0 ep) vs 540@136 (4.0 ep)  - does more data help under
     the standard 4-epoch recipe? (compute differs by 2x, exposure per example matched)
  B. STEP-MATCHED   1080@272 vs 540@272                     - fixed budget, more data
  C. BUDGET EFFECT  1080@272 vs 1080@136                    - same corpus, 2x compute
"""
import glob, json, os, pathlib, re, statistics
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/outputs/multi_agent_bert"
SEEDS = [42, 43, 44]
LABELS = ["business", "education", "health", "shopping", "medical",
          "sports", "tech", "finance", "social"]

CELLS = {
    ("Topic-540", 136): OUT / "experiment_topic_matched_compute" / "mc540_seed{s}",
    ("Topic-540", 272): OUT / "experiment_topic540_272steps" / "mc540x2_seed{s}",
    ("Topic-1080", 136): OUT / "experiment_topic1080_136steps" / "mc1080_seed{s}",
    ("Topic-1080", 272): OUT / "experiment_topic1080_272steps" / "mc1080x2_seed{s}",
}


def analyse(d):
    pj = glob.glob(os.path.join(str(d), "*_predictions.json"))
    if not pj:
        raise SystemExit(f"missing predictions: {d}")
    rows = json.load(open(pj[0], encoding="utf-8"))
    n = len(rows)
    f1s, sup = {}, Counter(r["true_label"] for r in rows)
    for lab in LABELS:
        tp = sum(1 for r in rows if r["true_label"] == lab and r["predicted_label"] == lab)
        fp = sum(1 for r in rows if r["true_label"] != lab and r["predicted_label"] == lab)
        fn = sum(1 for r in rows if r["true_label"] == lab and r["predicted_label"] != lab)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s[lab] = 2 * p * rc / (p + rc) if p + rc else 0.0
    pred = Counter(r["predicted_label"] for r in rows)
    top_lab, top_n = pred.most_common(1)[0]
    return {"acc": sum(r["correct"] for r in rows) / n,
            "macro_f1": sum(f1s.values()) / len(LABELS),
            "weighted_f1": sum(f1s[l] * sup[l] for l in LABELS) / n,
            "per_class_f1": f1s, "n_pred": len(pred),
            "top_label": top_lab, "top_share": top_n / n}


def steps(d):
    log = pathlib.Path(d) / "finetune.log"
    if not log.exists():
        return "?", "?"
    t = log.read_text(encoding="utf-8", errors="replace")
    bars = re.findall(r"\|\s*(\d+)/(\d+)\s*\[", t)
    ep = re.findall(r"'epoch': ([0-9.]+)}", t)
    return ("/".join(bars[-1]) if bars else "?"), (ep[-1] if ep else "?")


R = {(c, b, s): analyse(pathlib.Path(str(p).format(s=s)))
     for (c, b), p in CELLS.items() for s in SEEDS}
S = {(c, b, s): steps(pathlib.Path(str(p).format(s=s)))
     for (c, b), p in CELLS.items() for s in SEEDS}

lines = []


def out(x=""):
    print(x)
    lines.append(x)


def m(c, b, k):
    xs = [R[(c, b, s)][k] for s in SEEDS]
    return statistics.mean(xs), statistics.pstdev(xs)


out("# Generated topic corpora: corpus x budget 2x2 on Silver-1163")
out()
out("One recipe throughout; only the training corpus and --max_steps differ.")
out("primary_only evaluation, Silver used for final evaluation only. 3 seeds, no")
out("significance claimed.")
out()
out("## Accuracy grid (mean +/- SD over seeds 42/43/44)")
out()
out("| corpus | 136 steps | 272 steps |")
out("|---|---|---|")
for c in ("Topic-540", "Topic-1080"):
    a1, s1 = m(c, 136, "acc")
    a2, s2 = m(c, 272, "acc")
    out(f"| {c} | {a1:.4f} +/- {s1:.4f} | {a2:.4f} +/- {s2:.4f} |")
out()
out("Epochs per cell: 540@136 = 4.0 | 540@272 = 8.0 | 1080@136 = 2.0 | 1080@272 = 4.0")

out()
out("## Per-run detail")
out()
out("| corpus | budget | seed | steps | epoch | acc | macro-F1 | weighted-F1 | #classes | top share |")
out("|---|--:|--:|---|--:|--:|--:|--:|--:|--:|")
for c in ("Topic-540", "Topic-1080"):
    for b in (136, 272):
        for s in SEEDS:
            r, (st, ep) = R[(c, b, s)], S[(c, b, s)]
            out(f"| {c} | {b} | {s} | {st} | {ep} | {r['acc']:.4f} | {r['macro_f1']:.4f} "
                f"| {r['weighted_f1']:.4f} | {r['n_pred']} | {r['top_share']:.1%} |")

out()
out("## Mean +/- SD, all metrics")
out()
out("| cell | acc | macro-F1 | weighted-F1 |")
out("|---|---|---|---|")
for c in ("Topic-540", "Topic-1080"):
    for b in (136, 272):
        cols = [f"{m(c, b, k)[0]:.4f} +/- {m(c, b, k)[1]:.4f}"
                for k in ("acc", "macro_f1", "weighted_f1")]
        out(f"| {c} @ {b} | {cols[0]} | {cols[1]} | {cols[2]} |")

CONTRASTS = [
    ("A. EPOCH-MATCHED: 1080@272 (4.0 ep) - 540@136 (4.0 ep)", ("Topic-1080", 272), ("Topic-540", 136)),
    ("B. STEP-MATCHED: 1080@272 - 540@272", ("Topic-1080", 272), ("Topic-540", 272)),
    ("C. BUDGET: 1080@272 - 1080@136", ("Topic-1080", 272), ("Topic-1080", 136)),
]
for title, hi, lo in CONTRASTS:
    out()
    out(f"## {title}")
    out()
    out("| seed | d acc | d macro-F1 | d weighted-F1 |")
    out("|--:|--:|--:|--:|")
    d = {k: [] for k in ("acc", "macro_f1", "weighted_f1")}
    for s in SEEDS:
        a, b_ = R[(hi[0], hi[1], s)], R[(lo[0], lo[1], s)]
        for k in d:
            d[k].append(a[k] - b_[k])
        out(f"| {s} | {a['acc']-b_['acc']:+.4f} | {a['macro_f1']-b_['macro_f1']:+.4f} "
            f"| {a['weighted_f1']-b_['weighted_f1']:+.4f} |")
    out(f"| **mean** | **{statistics.mean(d['acc']):+.4f}** | "
        f"**{statistics.mean(d['macro_f1']):+.4f}** | **{statistics.mean(d['weighted_f1']):+.4f}** |")
    signs = {k: ("consistent" if len({x > 0 for x in d[k]}) == 1 else "MIXED") for k in d}
    out()
    out(f"- signs: acc {signs['acc']}, macro-F1 {signs['macro_f1']}, weighted-F1 {signs['weighted_f1']}")

out()
out("## Collapse check (all 12 runs)")
out()
bad = [f"{c}@{b}/s{s}" for (c, b, s) in R
       if R[(c, b, s)]["n_pred"] < 9 or R[(c, b, s)]["top_share"] >= 0.50]
out(f"- runs with <9 predicted classes or top share >=50%: {bad if bad else 'NONE'}")
out(f"- top-share range across all runs: {min(r['top_share'] for r in R.values()):.1%} - "
    f"{max(r['top_share'] for r in R.values()):.1%}")
out()
out("Reference: Original-180 corpus-draw sd at 136 steps = 0.0111 (9-run subset grid);")
out("Topic-180 grand mean 0.6190, Topic-360 0.6022.")

dst = OUT / "experiment_topic1080_272steps"
dst.mkdir(parents=True, exist_ok=True)
(dst / "RESULTS_CORPUS_BUDGET_2x2.md").write_text("\n".join(lines), encoding="utf-8")
json.dump({f"{c}@{b}_s{s}": R[(c, b, s)] for (c, b, s) in R},
          open(dst / "summary_2x2.json", "w", encoding="utf-8"), indent=2)
print(f"\nwrote {dst / 'RESULTS_CORPUS_BUDGET_2x2.md'}")
