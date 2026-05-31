"""
build_consolidated_annotation_sheet.py
======================================
Build ONE consolidated human-annotation sheet that supports three claims at once:
  (1) task-aware generation quality   (label/task correctness per task)
  (2) masking / per-sentence confirmation  (masked vs non-masked sentences)
  (3) CS validity + linguistic quality (code-switch validity, fluency, naturalness)

Source: the FRESH pre-refinement validation sample (refiner OFF). Sentences are
sampled balanced across tasks (topic/sentiment/NER) and across masked vs non-masked.

Masked = sentence score < acceptance_threshold while its scenario aggregate >=
acceptance_threshold (a weak sentence the scenario average would accept).
Threshold is READ FROM threshold_sweep.yaml (not hardcoded).

Outputs:
  experiments/outputs/switchlingua/human_eval/consolidated_annotation_sheet.csv

Note: predicted_or_generated_label and bio_tags_or_entities are left blank here
(the refiner-OFF sample has no task-validator output). Re-run with --task1-data
<jsonl> after Test 1 (validator ON) to fill them; the human columns do not need them.

Usage:
  python experiments/switchlingua/build_consolidated_annotation_sheet.py
  python experiments/switchlingua/build_consolidated_annotation_sheet.py --per-task 30 --seed 42
"""
import argparse
import csv
import json
import pathlib
import random
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG = pathlib.Path(__file__).parent / "threshold_sweep.yaml"
OUT_DIR = ROOT / "experiments" / "outputs" / "switchlingua" / "human_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import yaml  # noqa: E402

COLUMNS = [
    # ---- metadata (pre-filled) ----
    "sample_id", "source_experiment", "task", "text", "target_label",
    "predicted_or_generated_label", "bio_tags_or_entities",
    "pipeline_accepted", "pipeline_score", "masked_case", "annotator_id",
    # ---- human annotation (blank) ----
    "is_code_switched_yes_no", "label_or_task_correct_yes_no",
    "fluency_1_10", "naturalness_1_10", "overall_acceptable_yes_no",
    "error_type", "notes",
    # ---- NER-specific (blank; only fill for task=ner rows) ----
    "entities_correct_yes_no", "bio_valid_yes_no", "boundary_correct_yes_no",
]


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def target_label(rec):
    task = rec.get("task")
    if task == "sentiment":
        return str(rec.get("label", ""))
    if task == "topic":
        return str(rec.get("label", rec.get("topic", "")))
    if task == "ner":
        c = rec.get("task_constraints", {}) or {}
        ets = ",".join(c.get("entity_types", []) or [])
        must = ",".join(c.get("must_include_types", []) or [])
        return f"entity_types=[{ets}]; must=[{must}]; n={c.get('min_entities')}-{c.get('max_entities')}"
    return ""


def collect(records, threshold):
    """Return {task: {'masked': [...], 'control': [...]}} of sentence items."""
    buckets = {"topic": {"masked": [], "control": []},
               "sentiment": {"masked": [], "control": []},
               "ner": {"masked": [], "control": []}}
    for idx, rec in enumerate(records):
        scores = rec.get("sentence_scores")
        sents = rec.get("data_generation_result", [])
        if not isinstance(scores, list) or len(scores) < 2:
            continue
        task = rec.get("task", "topic")
        if task not in buckets:
            continue
        agg = statistics.mean(float(s) for s in scores)
        agg_accepts = agg >= threshold
        for j, txt in enumerate(sents):
            if j >= len(scores) or not isinstance(txt, str) or not txt.strip():
                continue
            score = float(scores[j])
            masked = (score < threshold) and agg_accepts
            item = {
                "scenario_index": idx, "task": task, "text": txt.strip(),
                "target_label": target_label(rec),
                "pipeline_score": round(score, 3),
                "pipeline_accepted": "yes" if score >= threshold else "no",
                "masked_case": "yes" if masked else "no",
                "scenario_aggregate": round(agg, 3),
            }
            if masked:
                buckets[task]["masked"].append(item)
            elif score >= threshold:
                buckets[task]["control"].append(item)
            # sentences below threshold in a rejected scenario are neither masked nor
            # high-scoring controls; we skip them to keep the masked/control contrast clean.
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=30, help="target rows per task (~3 tasks => ~90)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=str(CFG))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    threshold = float(cfg["acceptance_threshold"])
    data_path = ROOT / cfg["input"]["data"]
    if not data_path.exists():
        print(f"INPUT NOT FOUND: {data_path}\nGenerate the validation sample first.")
        return

    records = load_jsonl(data_path)
    buckets = collect(records, threshold)
    random.seed(args.seed)

    rows = []
    for task in ("topic", "sentiment", "ner"):
        masked = buckets[task]["masked"]
        control = buckets[task]["control"]
        random.shuffle(masked); random.shuffle(control)
        half = args.per_task // 2
        take_m = min(half, len(masked))
        take_c = min(args.per_task - take_m, len(control))
        # if one side is short, backfill from the other
        if take_m + take_c < args.per_task:
            take_m = min(len(masked), args.per_task - take_c)
        chosen = masked[:take_m] + control[:take_c]
        random.shuffle(chosen)
        rows.extend(chosen)
        print(f"  {task}: {take_m} masked + {take_c} control = {len(chosen)}  "
              f"(available masked={len(masked)}, control={len(control)})")

    random.shuffle(rows)
    sheet = OUT_DIR / "consolidated_annotation_sheet.csv"
    with open(sheet, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for i, it in enumerate(rows, 1):
            w.writerow({
                "sample_id": f"A{i:03d}",
                "source_experiment": "validation_raw (pre-refinement, refiner OFF)",
                "task": it["task"],
                "text": it["text"],
                "target_label": it["target_label"],
                "predicted_or_generated_label": "",   # fill after Test 1 (validator ON)
                "bio_tags_or_entities": "",            # fill after Test 1 for NER
                "pipeline_accepted": it["pipeline_accepted"],
                "pipeline_score": it["pipeline_score"],
                "masked_case": it["masked_case"],
                "annotator_id": "",
                # blank human columns
                "is_code_switched_yes_no": "", "label_or_task_correct_yes_no": "",
                "fluency_1_10": "", "naturalness_1_10": "", "overall_acceptable_yes_no": "",
                "error_type": "", "notes": "",
                "entities_correct_yes_no": "", "bio_valid_yes_no": "", "boundary_correct_yes_no": "",
            })

    n_masked = sum(1 for r in rows if r["masked_case"] == "yes")
    print(f"\nWrote {len(rows)} rows ({n_masked} masked, {len(rows)-n_masked} control) -> {sheet}")
    print(f"Threshold (from config) = {threshold}.  Fill predicted_label/bio after Test 1.")


if __name__ == "__main__":
    main()
