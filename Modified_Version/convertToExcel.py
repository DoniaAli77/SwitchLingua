import json
import pandas as pd

INPUT_JSONL = "./output/Arabic.jsonl"
OUTPUT_EXCEL = "./output/Arabic_analysis.xlsx"


def to_json_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def join_any(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(to_json_text(item) for item in value)
    if isinstance(value, dict):
        return to_json_text(value)
    return str(value)


def safe_dict(value):
    return value if isinstance(value, dict) else {}


rows = []

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        obj = json.loads(line)

        flu = safe_dict(obj.get("fluency_result"))
        nat = safe_dict(obj.get("naturalness_result"))
        soc = safe_dict(obj.get("social_cultural_result"))

        # New schema: list per sentence. Old schema fallback: single cs_ratio_result.
        cs_per_instance = obj.get("cs_ratio_results_per_instances")
        if not isinstance(cs_per_instance, list):
            cs_single = safe_dict(obj.get("cs_ratio_result"))
            cs_per_instance = [cs_single] if cs_single else []

        flu_per_instance = obj.get("fluency_results_per_instances")
        if not isinstance(flu_per_instance, list):
            flu_per_instance = []

        nat_per_instance = obj.get("naturalness_results_per_instances")
        if not isinstance(nat_per_instance, list):
            nat_per_instance = []

        soc_per_instance = obj.get("social_cultural_results_per_instances")
        if not isinstance(soc_per_instance, list):
            soc_per_instance = []

        ratio_scores = [
            item.get("ratio_score")
            for item in cs_per_instance
            if isinstance(item, dict) and isinstance(item.get("ratio_score"), (int, float))
        ]
        aggregate_cs_ratio_score = (
            round(sum(ratio_scores) / len(ratio_scores), 4) if ratio_scores else None
        )

        task_validation = safe_dict(obj.get("task_validation_result"))
        per_instance_validation = task_validation.get("per_instance_results")
        if not isinstance(per_instance_validation, list):
            per_instance_validation = []

        meta = {
            "task": obj.get("task"),
            "label": obj.get("label"),
            "topic": obj.get("topic"),
            "tense": obj.get("tense"),
            "perspective": obj.get("perspective"),
            "cs_ratio": obj.get("cs_ratio"),
            "gender": obj.get("gender"),
            "age": obj.get("age"),
            "education_level": obj.get("education_level"),
            "first_language": obj.get("first_language"),
            "second_language": obj.get("second_language"),
            "conversation_type": obj.get("conversation_type"),
            "cs_function": obj.get("cs_function"),
            "cs_type": obj.get("cs_type"),
            "score": obj.get("score"),
            "refine_count": obj.get("refine_count"),
            "task_constraints": to_json_text(obj.get("task_constraints")),
            "annotations": to_json_text(obj.get("annotations")),
        }

        aggregate_agent = {
            "fluency_score": flu.get("fluency_score"),
            "naturalness_score": nat.get("naturalness_score"),
            "aggregate_cs_ratio_score": aggregate_cs_ratio_score,
            "socio_cultural_score": soc.get("socio_cultural_score"),
            "fluency_errors": join_any(flu.get("errors")),
            "naturalness_observations": join_any(nat.get("observations")),
            "socio_cultural_issues": join_any(soc.get("issues")),
            "fluency_summary": flu.get("summary"),
            "naturalness_summary": nat.get("summary"),
            "socio_cultural_summary": soc.get("summary"),
            "task_validation_passed": task_validation.get("passed"),
            "task_validation_confidence": task_validation.get("confidence"),
            "task_validation_predicted_label": task_validation.get("predicted_label"),
            "task_validation_notes": task_validation.get("notes"),
            "task_validation_errors": join_any(task_validation.get("errors")),
        }

        sents = obj.get("data_generation_result")
        if not isinstance(sents, list):
            sents = [sents] if sents is not None else []
        if not sents:
            sents = [""]

        for i, sent in enumerate(sents):
            cs_instance = cs_per_instance[i] if i < len(cs_per_instance) and isinstance(cs_per_instance[i], dict) else {}
            val_instance = (
                per_instance_validation[i]
                if i < len(per_instance_validation) and isinstance(per_instance_validation[i], dict)
                else {}
            )
            flu_inst = flu_per_instance[i] if i < len(flu_per_instance) and isinstance(flu_per_instance[i], dict) else {}
            nat_inst = nat_per_instance[i] if i < len(nat_per_instance) and isinstance(nat_per_instance[i], dict) else {}
            soc_inst = soc_per_instance[i] if i < len(soc_per_instance) and isinstance(soc_per_instance[i], dict) else {}

            sentence_level = {
                "sentence_index": i,
                "text": sent,
                "sentence_cs_ratio_score": cs_instance.get("ratio_score"),
                "sentence_cs_ratio_computed": cs_instance.get("computed_ratio"),
                "sentence_cs_ratio_notes": cs_instance.get("notes"),
                # Backfill from aggregate results when sentence-level fields are absent.
                "sentence_fluency_score": flu_inst.get("fluency_score", flu.get("fluency_score")),
                "sentence_fluency_errors": join_any(flu_inst.get("errors", flu.get("errors"))),
                "sentence_fluency_summary": flu_inst.get("summary", flu.get("summary")),
                "sentence_naturalness_score": nat_inst.get("naturalness_score", nat.get("naturalness_score")),
                "sentence_naturalness_observations": join_any(
                    nat_inst.get("observations", nat.get("observations"))
                ),
                "sentence_naturalness_summary": nat_inst.get("summary", nat.get("summary")),
                "sentence_socio_cultural_score": soc_inst.get(
                    "socio_cultural_score", soc.get("socio_cultural_score")
                ),
                "sentence_socio_cultural_issues": join_any(
                    soc_inst.get("issues", soc.get("issues"))
                ),
                "sentence_socio_cultural_summary": soc_inst.get("summary", soc.get("summary")),
                "sentence_validation_passed": val_instance.get("passed", task_validation.get("passed")),
                "sentence_validation_confidence": val_instance.get(
                    "confidence", task_validation.get("confidence")
                ),
                "sentence_validation_predicted_label": val_instance.get(
                    "predicted_label", task_validation.get("predicted_label")
                ),
                "sentence_validation_notes": val_instance.get("notes", task_validation.get("notes")),
                "sentence_validation_errors": join_any(
                    val_instance.get("errors", task_validation.get("errors"))
                ),
            }

            rows.append({**meta, **aggregate_agent, **sentence_level})

df = pd.DataFrame(rows)

preferred = [
    "task",
    "label",
    "topic",
    "text",
    "score",
    "fluency_score",
    "naturalness_score",
    "aggregate_cs_ratio_score",
    "socio_cultural_score",
    "sentence_cs_ratio_score",
    "sentence_cs_ratio_computed",
    "sentence_cs_ratio_notes",
    "sentence_fluency_score",
    "sentence_fluency_errors",
    "sentence_fluency_summary",
    "sentence_naturalness_score",
    "sentence_naturalness_observations",
    "sentence_naturalness_summary",
    "sentence_socio_cultural_score",
    "sentence_socio_cultural_issues",
    "sentence_socio_cultural_summary",
    "task_validation_passed",
    "task_validation_confidence",
    "task_validation_predicted_label",
    "task_validation_notes",
    "task_validation_errors",
    "sentence_validation_passed",
    "sentence_validation_confidence",
    "sentence_validation_predicted_label",
    "sentence_validation_notes",
    "sentence_validation_errors",
    "task_constraints",
    "annotations",
    "tense",
    "perspective",
    "conversation_type",
    "cs_ratio",
    "cs_type",
    "cs_function",
    "gender",
    "age",
    "education_level",
    "first_language",
    "second_language",
    "refine_count",
    "sentence_index",
    "fluency_errors",
    "naturalness_observations",
    "socio_cultural_issues",
    "fluency_summary",
    "naturalness_summary",
    "socio_cultural_summary",
]

ordered = [c for c in preferred if c in df.columns]
df = df[ordered + [c for c in df.columns if c not in ordered]]

df.to_excel(OUTPUT_EXCEL, index=False)
print(f"Saved {len(df)} rows to {OUTPUT_EXCEL}")