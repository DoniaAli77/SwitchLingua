"""One-off analysis of the silver-1044 x Topic-540 primary_only inference runs.

Reads the frozen silver subset (for cs_type/silver_topic) and the two
predictions.csv files (xlmr, mbert) written by evaluate_pipeline.py, and
prints all the extra tables the handover asked for that evaluate_pipeline.py
doesn't compute on its own: weighted-F1, confusion matrix, predicted-vs-silver
distribution, tag-vs-intrasentential breakdown, confidence distribution and
% below 0.90.
"""
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).parent.parent
OUT = ROOT / "experiments/outputs/multi_agent_bert/experiment_silver_topic540"
FROZEN = OUT / "silver_primary_1044.jsonl"

LABELS = ["business", "education", "health", "shopping", "medical",
          "sports", "tech", "finance", "social"]

frozen_rows = [json.loads(l) for l in open(FROZEN, encoding="utf-8")]
cs_types = [r["cs_type"] for r in frozen_rows]
silver_topics = [r["silver_topic"] for r in frozen_rows]

RUNS = {
    "XLM-R": OUT / "xlmr/silver1044_xlmr__full_pipeline_predictions.csv",
    "mBERT": OUT / "mbert/silver1044_mbert__full_pipeline_predictions.csv",
}

def load_preds(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows

for name, path in RUNS.items():
    rows = load_preds(path)
    assert len(rows) == len(frozen_rows), (name, len(rows), len(frozen_rows))
    y_true = [r["true_label"] for r in rows]
    y_pred = [r["predicted_label"] for r in rows]
    conf = [float(r["confidence"]) for r in rows]

    print(f"\n{'='*70}\n{name}\n{'='*70}")

    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, labels=LABELS, average="weighted")
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    print(f"accuracy={acc:.4f}  macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    print("\nConfusion matrix (rows=true silver, cols=predicted):")
    header = "true\\pred".ljust(10) + "".join(l[:4].rjust(6) for l in LABELS)
    print(header)
    for lbl, row in zip(LABELS, cm):
        print(lbl.ljust(10) + "".join(str(v).rjust(6) for v in row))

    # Predicted vs silver distribution
    true_dist = Counter(y_true)
    pred_dist = Counter(y_pred)
    print("\nlabel        silver_n   pred_n")
    for lbl in LABELS:
        print(f"{lbl:<12} {true_dist.get(lbl,0):>8} {pred_dist.get(lbl,0):>8}")

    # Tag vs intrasentential breakdown
    print("\ncs_type breakdown:")
    for cst in ["tag", "intrasentential"]:
        idx = [i for i, c in enumerate(cs_types) if c == cst]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        a = sum(t == p for t, p in zip(yt, yp)) / len(yt)
        mf1 = f1_score(yt, yp, labels=LABELS, average="macro")
        print(f"  {cst:<16} n={len(idx):>5}  acc={a:.4f}  macro_f1={mf1:.4f}")

    # Confidence distribution
    import statistics as st
    below = sum(1 for c in conf if c < 0.90)
    print(f"\nconfidence: mean={st.mean(conf):.4f} median={st.median(conf):.4f} "
          f"stdev={st.pstdev(conf):.4f} min={min(conf):.4f} max={max(conf):.4f}")
    print(f"% below 0.90: {below/len(conf)*100:.2f}%  ({below}/{len(conf)})")
    # quartile buckets
    buckets = [(0.0,0.3),(0.3,0.5),(0.5,0.7),(0.7,0.9),(0.9,1.01)]
    for lo, hi in buckets:
        n = sum(1 for c in conf if lo <= c < hi)
        print(f"  [{lo:.1f},{hi:.1f}): {n:>5} ({n/len(conf)*100:5.1f}%)")

    # Save per-class P/R/F1/support for reference (weighted avg support = macro support)
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    print("\nper-class:")
    for lbl, pp, rr, ff, ss in zip(LABELS, p, r, f1, support):
        print(f"  {lbl:<12} P={pp:.4f} R={rr:.4f} F1={ff:.4f} support={ss}")
