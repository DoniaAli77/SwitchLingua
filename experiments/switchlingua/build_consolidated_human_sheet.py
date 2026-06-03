"""
build_consolidated_human_sheet.py — assemble ONE human-annotation sheet (80-100 rows) from
existing saved data to validate the remaining LLM-judged claims. No generation; no NER/prompt change.

Sources:
  task_aware_eval/task_aware_details.jsonl  (topic/sentiment/NER + blind-judge label + fluency/naturalness)
  task_validator/validator_verdicts.jsonl   (real TaskValidator pass/fail, joined by text)
  per_sentence/validation_raw/Arabic.jsonl   (quality score, task_constraints, masked/control, per-instance)

Output: experiments/outputs/switchlingua/human_eval/consolidated_human_annotation_sheet.csv
"""
import csv, json, pathlib, random, statistics, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
if str(MODIFIED_CORE) not in sys.path:
    sys.path.insert(0, str(MODIFIED_CORE))
import yaml
from utils import compute_true_cs_stats

B = ROOT / "experiments" / "outputs" / "switchlingua"
OUT = B / "human_eval"; OUT.mkdir(parents=True, exist_ok=True)
THRESHOLD = float(yaml.safe_load(open(pathlib.Path(__file__).parent / "threshold_sweep.yaml", encoding="utf-8"))["acceptance_threshold"])
SEED = 7

COLS = ["sample_id", "source_experiment", "task", "text", "target_label", "task_constraints",
        "pipeline_task_correct_or_judge_label", "task_validator_passed", "quality_score",
        "fluency", "naturalness", "cs_valid_deterministic", "cs_ratio_deterministic", "masked_case",
        "notes_for_annotator",
        "human_task_correct", "human_cs_valid", "human_fluency_1_5", "human_naturalness_1_5",
        "human_overall_acceptable", "human_error_type", "human_notes",
        "human_sentiment_label",
        "human_entities_present", "required_entity_types_present", "required_entities_english_script", "human_ner_correct",
        "human_arabic_token_count", "human_english_token_count", "human_other_token_count"]


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def main():
    details = load(B / "task_aware_eval" / "task_aware_details.jsonl")
    verdicts = {v["text"].strip(): v for v in load(B / "task_validator" / "validator_verdicts.jsonl")}
    raw = load(B / "per_sentence" / "validation_raw" / "Arabic.jsonl")

    # maps from validation_raw, keyed by sentence text
    qmap, recmap, masked_map, flu_map, nat_map = {}, {}, {}, {}, {}
    for r in raw:
        scores = r.get("sentence_scores", []) or []
        sents = r.get("data_generation_result", []) or []
        flu = r.get("fluency_results_per_instances", []) or []
        nat = r.get("naturalness_results_per_instances", []) or []
        agg = statistics.mean([float(s) for s in scores]) if scores else 0.0
        for j, t in enumerate(sents):
            if not isinstance(t, str):
                continue
            k = t.strip()
            if j < len(scores):
                sc = float(scores[j]); qmap[k] = round(sc, 3)
                masked_map[k] = ("masked" if (sc < THRESHOLD and agg >= THRESHOLD) else ("control" if sc >= THRESHOLD else "low_rejected"))
            recmap[k] = r
            if j < len(flu) and isinstance(flu[j], dict): flu_map[k] = flu[j].get("fluency_score")
            if j < len(nat) and isinstance(nat[j], dict): nat_map[k] = nat[j].get("naturalness_score")

    def constraints_str(rec, task):
        c = rec.get("task_constraints", {}) if rec else {}
        if task == "ner":
            return f"entity_types={c.get('entity_types')}; must={c.get('must_include_types')}; n={c.get('min_entities')}-{c.get('max_entities')}; script={c.get('target_entities_script','english')}"
        if task == "sentiment":
            return f"intensity={c.get('intensity')}; ambiguity={c.get('ambiguity')}"
        return ""

    rows = []
    used = set()

    def add_taskaware(d, note):
        t = d["text"].strip()
        if t in used:
            return
        used.add(t)
        task = d["task"]
        rec = recmap.get(t)
        if task == "sentiment":
            judge = str(d.get("predicted", ""))
        elif task == "topic":
            judge = "relevant" if d.get("topic_relevant") else "not_relevant"
        else:
            judge = "pass" if d.get("ner_passed") else "fail"
        v = verdicts.get(t, {})
        rows.append({
            "source_experiment": "task_aware", "task": task, "text": t,
            "target_label": d.get("target_label", ""), "task_constraints": constraints_str(rec, task),
            "pipeline_task_correct_or_judge_label": judge,
            "task_validator_passed": ("yes" if v.get("validator_passed") else ("no" if v.get("validator_passed") is False else "")),
            "quality_score": qmap.get(t, ""), "fluency": d.get("fluency", ""), "naturalness": d.get("naturalness", ""),
            "cs_valid_deterministic": ("yes" if d.get("is_code_switched") else "no"),
            "cs_ratio_deterministic": d.get("cs_ar_ratio", ""),
            "masked_case": "yes" if masked_map.get(t) == "masked" else "no",
            "notes_for_annotator": note,
        })

    sent = [d for d in details if d["task"] == "sentiment"]
    topic = [d for d in details if d["task"] == "topic"]
    ner = [d for d in details if d["task"] == "ner"]
    random.seed(SEED)
    for L in (sent, topic, ner):
        random.shuffle(L)

    # sentiment: all 11 disputed neutral + ~13 others
    disputed = [d for d in sent if str(d.get("predicted", "")).lower() != str(d.get("target_label", "")).lower()]
    agreed = [d for d in sent if d not in disputed]
    for d in disputed:
        add_taskaware(d, "SENTIMENT-DISPUTED: blind judge disagreed with the target sentiment - resolve the true label")
    for d in agreed[:13]:
        add_taskaware(d, "sentiment: confirm the sentiment matches the target")
    # topic: 12
    for d in topic[:12]:
        add_taskaware(d, "topic: does the sentence belong to the target topic?")
    # NER: ~10 failed (missing) + ~10 accepted
    nfail = [d for d in ner if d.get("task_correct") is False]
    nacc = [d for d in ner if d.get("task_correct") is True]
    for d in nfail[:10]:
        add_taskaware(d, "NER-FAIL: judge says required English-script entity missing - verify entities")
    for d in nacc[:10]:
        add_taskaware(d, "NER-ACCEPTED: judge says entities satisfied - verify")

    # masking rows from validation_raw (dedupe vs task-aware)
    masked_rows = [(k, r) for k, r in masked_map.items() if r == "masked" and k not in used]
    control_rows = [(k, r) for k, r in masked_map.items() if r == "control" and k not in used]
    random.shuffle(masked_rows); random.shuffle(control_rows)

    def add_masking(t, status):
        if t in used:
            return
        used.add(t)
        rec = recmap.get(t, {})
        task = rec.get("task", "?")
        st = compute_true_cs_stats(t)
        rows.append({
            "source_experiment": "masking", "task": task, "text": t,
            "target_label": rec.get("label", rec.get("topic", "")), "task_constraints": constraints_str(rec, task),
            "pipeline_task_correct_or_judge_label": "", "task_validator_passed": "",
            "quality_score": qmap.get(t, ""), "fluency": flu_map.get(t, ""), "naturalness": nat_map.get(t, ""),
            "cs_valid_deterministic": "yes" if st.get("is_code_switched") else "no",
            "cs_ratio_deterministic": st.get("cs_ar_ratio", ""),
            "masked_case": "yes" if status == "masked" else "no",
            "notes_for_annotator": ("MASKED weak sentence (aggregate hid it)" if status == "masked"
                                    else "high-scoring control sentence"),
        })
    for k, _ in masked_rows[:15]:
        add_masking(k, "masked")
    for k, _ in control_rows[:15]:
        add_masking(k, "control")

    # shuffle final, assign ids, mark a CS-ratio subset (~12) for manual token counting
    random.shuffle(rows)
    cs_subset = set(list(range(len(rows)))[:: max(1, len(rows) // 12)][:12])
    for i, r in enumerate(rows, 1):
        r["sample_id"] = f"H{i:03d}"
        if (i - 1) in cs_subset:
            r["notes_for_annotator"] = "[CS-RATIO] " + r["notes_for_annotator"] + " - also count Arabic/English/other tokens"
        for c in COLS:
            r.setdefault(c, "")

    with open(OUT / "consolidated_human_annotation_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})

    from collections import Counter
    print(f"rows: {len(rows)}  by source: {dict(Counter(r['source_experiment'] for r in rows))}  by task: {dict(Counter(r['task'] for r in rows))}")
    print(f"masked: {sum(1 for r in rows if r['masked_case']=='yes')} | sentiment-disputed: {sum(1 for r in rows if 'DISPUTED' in r['notes_for_annotator'])} | CS-ratio subset: {sum(1 for r in rows if '[CS-RATIO]' in r['notes_for_annotator'])}")
    print(f"-> {OUT/'consolidated_human_annotation_sheet.csv'}")


if __name__ == "__main__":
    main()
