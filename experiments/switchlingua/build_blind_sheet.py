"""
build_blind_sheet.py — split the full consolidated sheet into a BLIND annotation sheet + a KEY.
The BLIND sheet hides everything that could bias the annotator (AI label, validator, scores,
deterministic CS, masked_case, AI-revealing notes). The KEY holds the hidden metadata and merges
back by sample_id. No generation; no NER/prompt change.

Outputs (experiments/outputs/switchlingua/human_eval/):
  consolidated_human_annotation_sheet_BLIND.csv
  consolidated_human_annotation_key.csv
"""
import csv, pathlib

HD = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "outputs" / "switchlingua" / "human_eval"
FULL = HD / "consolidated_human_annotation_sheet.csv"

HUMAN_COLS = ["human_task_correct", "human_cs_valid", "human_fluency_1_5", "human_naturalness_1_5",
              "human_overall_acceptable", "human_error_type", "human_notes", "human_sentiment_label",
              "human_entities_present", "required_entity_types_present", "required_entities_english_script",
              "human_ner_correct", "human_arabic_token_count", "human_english_token_count", "human_other_token_count"]
# annotator needs: id, task, target_label, task_constraints (NER only), text, sanitized note, + human cols
BLIND_COLS = ["sample_id", "task", "target_label", "task_constraints", "text", "notes_for_annotator"] + HUMAN_COLS
# everything hidden (merges back by sample_id)
KEY_COLS = ["sample_id", "source_experiment", "task", "target_label",
            "pipeline_task_correct_or_judge_label", "task_validator_passed", "quality_score",
            "fluency", "naturalness", "cs_valid_deterministic", "cs_ratio_deterministic",
            "masked_case", "original_notes_for_annotator"]


def blind_note(r):
    """Sanitized hint: keep the [CS-RATIO] marker, drop anything that reveals an AI/masked decision."""
    cs = "[CS-RATIO] " if "[CS-RATIO]" in (r.get("notes_for_annotator") or "") else ""
    t = r["task"]
    base = {"sentiment": "rate the sentiment label and overall quality",
            "topic": "does the sentence belong to the target topic?",
            "ner": "check the required entities (see task_constraints): present? English/Latin script? count in range?"}.get(t, "rate the sentence")
    if cs:
        base += " - also count Arabic / English / other tokens"
    return cs + base


def main():
    rows = list(csv.DictReader(FULL.open(encoding="utf-8-sig")))

    with open(HD / "consolidated_human_annotation_sheet_BLIND.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=BLIND_COLS); w.writeheader()
        for r in rows:
            out = {c: "" for c in BLIND_COLS}
            out["sample_id"] = r["sample_id"]
            out["task"] = r["task"]
            out["target_label"] = r["target_label"]
            # task_constraints only for NER (annotators need the entity policy); blank otherwise
            out["task_constraints"] = r["task_constraints"] if r["task"] == "ner" else ""
            out["text"] = r["text"]
            out["notes_for_annotator"] = blind_note(r)
            # human cols left blank
            w.writerow(out)

    with open(HD / "consolidated_human_annotation_key.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=KEY_COLS); w.writeheader()
        for r in rows:
            w.writerow({
                "sample_id": r["sample_id"], "source_experiment": r["source_experiment"], "task": r["task"],
                "target_label": r["target_label"],
                "pipeline_task_correct_or_judge_label": r["pipeline_task_correct_or_judge_label"],
                "task_validator_passed": r["task_validator_passed"], "quality_score": r["quality_score"],
                "fluency": r["fluency"], "naturalness": r["naturalness"],
                "cs_valid_deterministic": r["cs_valid_deterministic"], "cs_ratio_deterministic": r["cs_ratio_deterministic"],
                "masked_case": r["masked_case"], "original_notes_for_annotator": r["notes_for_annotator"],
            })

    print(f"BLIND sheet ({len(rows)} rows, {len(BLIND_COLS)} cols) + KEY ({len(KEY_COLS)} cols) written to {HD}")
    print("Give annotators only the BLIND sheet; keep the KEY hidden until analysis.")


if __name__ == "__main__":
    main()
