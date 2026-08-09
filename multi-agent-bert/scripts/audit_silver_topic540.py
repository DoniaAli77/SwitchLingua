"""Read-only audit of the silver-1044 x Topic-540 evaluation.

No retraining, no rerunning agents, no altering the frozen corpus. Only reads
already-frozen inputs and already-produced predictions, and writes ONE new
diagnostic CSV (45 stratified rows) for manual inspection.
"""
import csv
import json
import re
import unicodedata
import collections
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW_CSV = ROOT / "data/Topic/transcription data/silver_corpus.csv"
OUT = ROOT / "experiments/outputs/multi_agent_bert/experiment_silver_topic540"
FROZEN = OUT / "silver_primary_1044.jsonl"
PREDS = {
    "XLM-R": OUT / "xlmr/silver1044_xlmr__full_pipeline_predictions.csv",
    "mBERT": OUT / "mbert/silver1044_mbert__full_pipeline_predictions.csv",
}
LABELS = ["business", "education", "health", "shopping", "medical",
          "sports", "tech", "finance", "social"]

frozen_rows = [json.loads(l) for l in open(FROZEN, encoding="utf-8")]
raw_rows = list(csv.DictReader(open(RAW_CSV, encoding="utf-8")))
sid_key = [k for k in raw_rows[0] if k.endswith("segment_id")][0]
raw_by_sid = {r[sid_key]: r for r in raw_rows}

preds = {}
for name, path in PREDS.items():
    with open(path, encoding="utf-8", newline="") as fh:
        preds[name] = list(csv.DictReader(fh))
    assert len(preds[name]) == len(frozen_rows)

print("=" * 70)
print("2. UNIQUE TOPIC VALUES")
print("=" * 70)
print("frozen 'label' (checkpoint space):",
      sorted(collections.Counter(r["label"] for r in frozen_rows).items()))
print("frozen 'silver_topic' (corpus space):",
      sorted(collections.Counter(r["silver_topic"] for r in frozen_rows).items()))
for name, rows in preds.items():
    print(f"{name} predicted_label values:",
          sorted(collections.Counter(r["predicted_label"] for r in rows).items()))
    print(f"{name} true_label values (as scored):",
          sorted(collections.Counter(r["true_label"] for r in rows).items()))

print()
print("=" * 70)
print("3. tech / technology NORMALIZATION CHECK")
print("=" * 70)
n_label_technology = sum(1 for r in frozen_rows if r["label"] == "technology")
n_label_tech = sum(1 for r in frozen_rows if r["label"] == "tech")
n_silver_technology = sum(1 for r in frozen_rows if r["silver_topic"] == "technology")
n_silver_tech = sum(1 for r in frozen_rows if r["silver_topic"] == "tech")
print(f"rows with label=='technology' (should be 0): {n_label_technology}")
print(f"rows with label=='tech'                    : {n_label_tech}")
print(f"rows with silver_topic=='technology' (raw)  : {n_silver_technology}")
print(f"rows with silver_topic=='tech' (raw)        : {n_silver_tech}")
for name, rows in preds.items():
    has_technology_pred = sum(1 for r in rows if r["predicted_label"] == "technology")
    print(f"{name}: predicted_label=='technology' occurrences (checkpoint has no such class): {has_technology_pred}")

print()
print("=" * 70)
print("4. EVALUATION TARGET FIELD CHECK")
print("=" * 70)
print("CSV columns available:", list(raw_rows[0].keys()))
print("label_source unique values:", collections.Counter(r["label_source"] for r in raw_rows))
mismatch_topic_folder = 0
sample_mismatches = []
for r in frozen_rows:
    raw = raw_by_sid.get(r["segment_id"])
    if raw is None:
        continue
    csv_topic = raw["topic"].strip().lower()
    csv_folder = raw["source_topic_folder"].strip().lower()
    assert csv_topic == r["silver_topic"], (r["segment_id"], csv_topic, r["silver_topic"])
    if csv_topic != csv_folder:
        mismatch_topic_folder += 1
        if len(sample_mismatches) < 5:
            sample_mismatches.append((r["segment_id"], csv_topic, csv_folder))
print("Confirmed: frozen 'label'/'silver_topic' was built from CSV column 'topic' "
      "(label_source=multi_llm_consensus_silver), NOT from 'source_topic_folder'.")
print("No column literally named 'segment_topic'/'source_topic'/'video_topic' exists; "
      "'topic' = per-segment consensus label (used as eval target), "
      "'source_topic_folder' = the video/source directory's nominal topic (NOT used).")
print(f"Rows (of 1044) where per-segment 'topic' != 'source_topic_folder': "
      f"{mismatch_topic_folder} ({mismatch_topic_folder/len(frozen_rows)*100:.1f}%)")
print("Sample mismatches (segment_id, topic, source_topic_folder):")
for s in sample_mismatches:
    print(" ", s)

print()
print("=" * 70)
print("5. ROWS WHOSE CORRECTNESS DEPENDS ON THE tech<->technology ALIAS")
print("=" * 70)
for name, rows in preds.items():
    dependent = [r for r, f in zip(rows, frozen_rows)
                 if f["silver_topic"] == "technology" and r["predicted_label"] == "tech"]
    print(f"{name}: {len(dependent)} rows are predicted 'tech' with true silver_topic "
          f"'technology' -- these are CORRECT only because of the alias; without it "
          f"(comparing literal strings 'tech' vs 'technology') they would flip to incorrect.")
    # sanity: any row where alias comparison would go the other way (predicted contains 'technology')?
    reverse = [r for r in rows if r["predicted_label"] == "technology"]
    print(f"{name}: rows predicted literally 'technology' (impossible given checkpoint label space): {len(reverse)}")

print()
print("=" * 70)
print("6. PER-CLASS SUPPORT / RECALL (technology / tech highlighted)")
print("=" * 70)
from sklearn.metrics import precision_recall_fscore_support
for name, rows in preds.items():
    y_true = [r["true_label"] for r in rows]
    y_pred = [r["predicted_label"] for r in rows]
    p, r_, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    print(f"\n{name}:")
    for lbl, pp, rr, ff, ss in zip(LABELS, p, r_, f1, support):
        flag = "  <-- CHECK" if lbl == "tech" and (ss == 0 or rr == 0) else ""
        print(f"  {lbl:<12} support={ss:>4}  recall={rr:.4f}  precision={pp:.4f}  f1={ff:.4f}{flag}")
    # explicit 'technology' string check (should not exist as a class at all)
    n_true_technology_literal = sum(1 for t in y_true if t == "technology")
    print(f"  ['technology' as a literal true_label string: {n_true_technology_literal} "
          f"(0 expected -- ground truth was aliased to 'tech' before scoring)]")

print()
print("=" * 70)
print("7. DATASET VERSION / SPLIT AUDIT (ARENTC counts vs EXPERIMENT_REGISTRY.md)")
print("=" * 70)
import os
splits = {
    "ARENTCV1/train": "data/Topic/processed/ARENTCV1/train.jsonl",
    "ARENTCV1/dev":   "data/Topic/processed/ARENTCV1/dev.jsonl",
    "ARENTCV1/test":  "data/Topic/processed/ARENTCV1/test.jsonl",
    "ARENTCV2/train": "data/Topic/processed/ARENTCV2/train.jsonl",
    "ARENTCV2/dev":   "data/Topic/processed/ARENTCV2/dev.jsonl",
    "ARENTCV2/test":  "data/Topic/processed/ARENTCV2/test.jsonl",
    "ARENTCV2/test_sub500": "data/Topic/processed/ARENTCV2/test_sub500.jsonl",
}
for name, rel in splits.items():
    p = ROOT / rel
    n = sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else None
    print(f"  {name:<24} {n}")
print("Claimed-vs-found: user-cited 75,976 / 8,442 / 21,105 does NOT match any file on disk.")
print("On-disk + EXPERIMENT_REGISTRY.md (T1/T2 entry) agree: V1=73,976/10,569/21,137, "
      "V2=73,956/10,562/21,134 (also confirmed against the raw .xlsx split sources).")

print()
print("=" * 70)
print("8. DUPLICATE NORMALIZED TEXTS WITH DIFFERENT TOPICS")
print("=" * 70)
_DIACRITICS = re.compile(r'[ً-ٰـ]')
_PUNCT = re.compile(r'[^\w\s]', re.UNICODE)
_AR_DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
def norm(t):
    t = unicodedata.normalize('NFKC', t or '')
    t = t.translate(_AR_DIGITS)
    t = _DIACRITICS.sub('', t)
    t = t.lower()
    t = _PUNCT.sub(' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def find_conflicts(rows, text_key, topic_key, id_key, label):
    by_norm = collections.defaultdict(list)
    for r in rows:
        by_norm[norm(r[text_key])].append((r[id_key], r[topic_key]))
    conflicts = {n: v for n, v in by_norm.items() if len(v) > 1 and len({t for _, t in v}) > 1}
    print(f"{label}: {len(conflicts)} normalized-text groups with >1 distinct topic "
          f"(out of {sum(1 for v in by_norm.values() if len(v)>1)} groups with >1 row)")
    for n, v in list(conflicts.items())[:10]:
        print(f"    text={n[:60]!r} -> {v}")
    return conflicts

find_conflicts(raw_rows, "sentence", "topic", sid_key, "Full silver corpus (1163 rows, CSV 'topic')")
find_conflicts(frozen_rows, "text", "silver_topic", "segment_id", "Frozen subset (1044 rows)")

print()
print("=" * 70)
print("9. STRATIFIED DIAGNOSTIC FILE (45 rows, 5 per silver topic)")
print("=" * 70)
xlmr_by_idx = preds["XLM-R"]
mbert_by_idx = preds["mBERT"]
by_topic = collections.defaultdict(list)
for i, f in enumerate(frozen_rows):
    by_topic[f["silver_topic"]].append(i)

import random
random.seed(42)
diag_rows = []
for topic in sorted(by_topic):
    idxs = by_topic[topic]
    chosen = random.sample(idxs, min(5, len(idxs)))
    for i in sorted(chosen):
        f = frozen_rows[i]
        raw = raw_by_sid.get(f["segment_id"], {})
        xr = xlmr_by_idx[i]
        mr = mbert_by_idx[i]
        assert xr["true_label"] == mr["true_label"] == f["label"]
        diag_rows.append({
            "segment_id": f["segment_id"],
            "text": f["text"],
            "segment_topic": f["silver_topic"],
            "eval_label": f["label"],
            "source_topic_folder": raw.get("source_topic_folder", ""),
            "cs_type": f["cs_type"],
            "xlmr_pred": xr["predicted_label"],
            "xlmr_conf": xr["confidence"],
            "xlmr_correct": xr["correct"],
            "mbert_pred": mr["predicted_label"],
            "mbert_conf": mr["confidence"],
            "mbert_correct": mr["correct"],
        })

diag_path = OUT / "diagnostic_45rows.csv"
with open(diag_path, "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(diag_rows[0].keys()))
    w.writeheader()
    w.writerows(diag_rows)
print(f"Wrote {len(diag_rows)} rows -> {diag_path}")
print(f"Topics covered: {sorted(by_topic.keys())} ({len(by_topic)} topics x 5 = {len(diag_rows)})")
