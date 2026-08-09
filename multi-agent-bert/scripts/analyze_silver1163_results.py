"""Combine the frozen 1,044-row silver eval with the new 119-row silver eval
into the full 1,163-row silver corpus result, per the two-column
silver_topic_test_1044.csv file (1,163 rows) supplied for this task.

Read-only over already-produced predictions: does NOT rerun the 1,044-row
inference (preserved exactly as-is) and does NOT rerun agentic mode. Writes
combined per-row predictions, aggregate metrics, confusion matrices, and a
1,044-vs-1,163 comparison summary.
"""
import csv
import json
import collections
from pathlib import Path

from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, confusion_matrix, accuracy_score,
)

ROOT = Path(__file__).parent.parent
OUT = ROOT / "experiments/outputs/multi_agent_bert/experiment_silver_topic540"
LABELS = ["business", "education", "health", "shopping", "medical",
          "sports", "tech", "finance", "social"]

MODELS = {
    "XLM-R": {
        "old": OUT / "xlmr/silver1044_xlmr__full_pipeline_predictions.csv",
        "new": OUT / "xlmr_new119/silvernew119_xlmr__full_pipeline_predictions.csv",
    },
    "mBERT": {
        "old": OUT / "mbert/silver1044_mbert__full_pipeline_predictions.csv",
        "new": OUT / "mbert_new119/silvernew119_mbert__full_pipeline_predictions.csv",
    },
}

new119_meta = [json.loads(l) for l in open(OUT / "silver_new119.jsonl", encoding="utf-8")]


def load(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def metrics_block(rows, label):
    y_true = [r["true_label"] for r in rows]
    y_pred = [r["predicted_label"] for r in rows]
    conf = [float(r["confidence"]) for r in rows]
    n = len(rows)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    print(f"\n--- {label}  (n={n}) ---")
    print(f"accuracy={acc:.4f}  macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}")

    p, r_, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    print("per-class:")
    for lbl, pp, rr, ff, ss in zip(LABELS, p, r_, f1, support):
        print(f"  {lbl:<12} P={pp:.4f} R={rr:.4f} F1={ff:.4f} support={ss}")

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    print("confusion matrix (rows=true, cols=pred):")
    print("            " + "".join(l[:4].rjust(6) for l in LABELS))
    for lbl, row in zip(LABELS, cm):
        print(f"  {lbl:<10}" + "".join(str(v).rjust(6) for v in row))

    true_dist = collections.Counter(y_true)
    pred_dist = collections.Counter(y_pred)
    print("true vs pred distribution:")
    for lbl in LABELS:
        print(f"  {lbl:<12} true={true_dist.get(lbl,0):>4}  pred={pred_dist.get(lbl,0):>4}")

    below90 = sum(1 for c in conf if c < 0.90)
    import statistics as st
    print(f"confidence: mean={st.mean(conf):.4f} median={st.median(conf):.4f} "
          f"min={min(conf):.4f} max={max(conf):.4f}  below_0.90={below90}/{n} "
          f"({below90/n*100:.2f}%)")

    return {
        "n": n, "accuracy": round(acc, 6), "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "per_class": [{"label": l, "precision": round(pp, 6), "recall": round(rr, 6),
                        "f1": round(ff, 6), "support": int(ss)}
                       for l, pp, rr, ff, ss in zip(LABELS, p, r_, f1, support)],
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        "true_distribution": dict(true_dist), "pred_distribution": dict(pred_dist),
        "confidence": {"mean": st.mean(conf), "median": st.median(conf),
                        "min": min(conf), "max": max(conf),
                        "pct_below_0.90": round(below90 / n * 100, 2)},
    }


summary = {}
for name, paths in MODELS.items():
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    old_rows = load(paths["old"])
    new_rows = load(paths["new"])
    combined_rows = old_rows + new_rows
    assert len(combined_rows) == 1163

    summary[name] = {}
    summary[name]["1044_original"] = metrics_block(old_rows, "1,044-row (ORIGINAL, unchanged)")
    summary[name]["119_new"] = metrics_block(new_rows, "119-row NEW diagnostic subset")
    summary[name]["1163_combined"] = metrics_block(combined_rows, "1,163-row COMBINED (full corpus)")

    # subsets of the 119 by cs_category / standalone
    print("\n--- 119-row breakdown by category ---")
    for cat_name, pred_fn in [
        ("named_entity_only (n=105)", lambda m: m["cs_category"] == "named_entity_only"),
        ("acronym_or_model_only (n=12)", lambda m: m["cs_category"] == "acronym_or_model_only"),
        ("standalone=no (n=2)", lambda m: m["standalone"] == "no"),
    ]:
        idx = [i for i, m in enumerate(new119_meta) if pred_fn(m)]
        sub_rows = [new_rows[i] for i in idx]
        block = metrics_block(sub_rows, cat_name)
        summary[name][cat_name] = block

    # write combined per-row predictions
    combined_path = OUT / f"{name.lower().replace('-', '')}_combined_1163_predictions.csv"
    with open(combined_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(combined_rows[0].keys()) + ["source_subset"])
        w.writeheader()
        for r in old_rows:
            row = dict(r); row["source_subset"] = "original_1044"
            w.writerow(row)
        for r, m in zip(new_rows, new119_meta):
            row = dict(r); row["source_subset"] = m["cs_category"] if m["standalone"] == "yes" else "standalone_no"
            w.writerow(row)
    print(f"\nwrote combined predictions -> {combined_path}")

with open(OUT / "silver1163_summary.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2)
print(f"\nwrote summary -> {OUT / 'silver1163_summary.json'}")
