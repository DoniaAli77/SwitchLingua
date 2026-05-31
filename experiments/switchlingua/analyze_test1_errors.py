"""
analyze_test1_errors.py — Error analysis for Test 1 (task-aware generation).
============================================================================
Diagnoses WHERE Test 1 failures come from (prompt vs model vs evaluator artifact).
Pure analysis of an existing details file — no API, no generation, no prompt changes.

Input : experiments/outputs/switchlingua/task_aware_eval/task_aware_details.jsonl
Source (for NER constraints): per_sentence/validation_raw/Arabic.jsonl

Computes:
  1. Sentiment: confusion matrix (target vs blind predicted), per-label accuracy, error examples
  2. NER: failure categories (missing required type / too few / too many / none), examples
  3. Quality mismatch: fluency>=8 & naturalness>=8 but task wrong; mean fluency/nat for correct vs wrong
  4. CS ratio: % within +/-10, +/-15, +/-20 of the 70% Arabic target; high-CS-validity but poor-ratio examples

Outputs:
  task_aware_error_analysis.md / .csv / .json   (in task_aware_eval/)
"""
import csv
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIR = ROOT / "experiments" / "outputs" / "switchlingua" / "task_aware_eval"
DETAILS = DIR / "task_aware_details.jsonl"
RAW = ROOT / "experiments" / "outputs" / "switchlingua" / "per_sentence" / "validation_raw" / "Arabic.jsonl"
CS_TARGET = 70.0  # Arabic (matrix) target ratio


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def ner_constraints():
    """Read NER min/max/must-include from the source data (not hardcoded)."""
    mn, mx, must = 2, 3, ["PER", "ORG"]
    if RAW.exists():
        for r in load(RAW):
            if r.get("task") == "ner":
                c = r.get("task_constraints", {}) or {}
                mn = int(c.get("min_entities", mn) or mn)
                mx = int(c.get("max_entities", mx) or mx)
                must = [str(t).strip().upper() for t in (c.get("must_include_types", must) or must)]
                break
    return mn, mx, must


def short(t, n=110):
    t = t.replace("\n", " ").strip()
    return t if len(t) <= n else t[:n] + "..."


def main():
    rows = load(DETAILS)
    sent = [r for r in rows if r["task"] == "sentiment"]
    ner = [r for r in rows if r["task"] == "ner"]
    mn, mx, must = ner_constraints()

    result = {"input": str(DETAILS), "ner_constraints": {"min": mn, "max": mx, "must_include": must}}

    # ---------- 1) SENTIMENT ----------
    labels = ["positive", "negative", "neutral"]
    preds_seen = sorted({(r.get("predicted") or "unknown") for r in sent})
    cols = labels + [p for p in preds_seen if p not in labels]
    confusion = {t: {p: 0 for p in cols} for t in labels}
    per_label = {t: {"n": 0, "correct": 0} for t in labels}
    sent_errors = []
    for r in sent:
        t = str(r.get("target_label", "")).strip().lower()
        p = str(r.get("predicted", "unknown")).strip().lower()
        if t in confusion:
            if p not in confusion[t]:
                confusion[t][p] = 0
            confusion[t][p] += 1
            per_label[t]["n"] += 1
            if t == p:
                per_label[t]["correct"] += 1
            else:
                sent_errors.append({"target": t, "predicted": p, "text": r["text"],
                                    "fluency": r.get("fluency"), "naturalness": r.get("naturalness")})
    sentiment_block = {
        "n": len(sent),
        "confusion_matrix": confusion,
        "per_label_accuracy_pct": {t: (round(100 * d["correct"] / d["n"], 1) if d["n"] else None)
                                   for t, d in per_label.items()},
        "n_errors": len(sent_errors),
    }
    result["sentiment"] = sentiment_block

    # ---------- 2) NER ----------
    cats = {"no_entities": 0, "too_few": 0, "too_many": 0}
    for t in must:
        cats[f"missing_{t}"] = 0
    ner_fail_examples = []
    ner_failed = 0
    for r in ner:
        counts = r.get("entity_counts", {}) or {}
        total = r.get("total_entities", sum(counts.values()) if counts else 0) or 0
        passed = bool(r.get("ner_passed"))
        flags = []
        if total == 0:
            flags.append("no_entities")
        if 0 < total < mn:
            flags.append("too_few")
        if total > mx:
            flags.append("too_many")
        for t in must:
            if counts.get(t, 0) == 0:
                flags.append(f"missing_{t}")
        if not passed:
            ner_failed += 1
            for fl in flags:
                cats[fl] = cats.get(fl, 0) + 1
            if len(ner_fail_examples) < 8:
                ner_fail_examples.append({"text": r["text"], "entity_counts": counts,
                                          "total_entities": total, "flags": flags})
    ner_block = {
        "n": len(ner), "n_failed": ner_failed,
        "failure_categories": cats,
        "note": f"required: {mn}-{mx} entities incl. {'+'.join(must)} (read from source data)",
    }
    result["ner"] = ner_block

    # ---------- 3) QUALITY MISMATCH ----------
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    correct = [r for r in rows if r.get("task_correct") is True]
    wrong = [r for r in rows if r.get("task_correct") is False]
    high_q_wrong = [r for r in wrong
                    if (num(r.get("fluency")) or 0) >= 8 and (num(r.get("naturalness")) or 0) >= 8]
    def mean_of(rs, k):
        vals = [num(r.get(k)) for r in rs]
        vals = [v for v in vals if v is not None]
        return round(statistics.mean(vals), 2) if vals else None
    quality_block = {
        "n_task_correct": len(correct), "n_task_wrong": len(wrong),
        "high_quality_but_wrong_count": len(high_q_wrong),
        "high_quality_but_wrong_pct_of_wrong": round(100 * len(high_q_wrong) / len(wrong), 1) if wrong else None,
        "mean_fluency_correct": mean_of(correct, "fluency"),
        "mean_fluency_wrong": mean_of(wrong, "fluency"),
        "mean_naturalness_correct": mean_of(correct, "naturalness"),
        "mean_naturalness_wrong": mean_of(wrong, "naturalness"),
        "examples": [{"task": r["task"], "target_label": r.get("target_label"),
                      "fluency": r.get("fluency"), "naturalness": r.get("naturalness"),
                      "text": r["text"]} for r in high_q_wrong[:8]],
    }
    result["quality_mismatch"] = quality_block

    # ---------- 4) CS RATIO ----------
    errs = []
    cs_examples = []
    for r in rows:
        ar = num(r.get("cs_ar_ratio"))
        if ar is None:
            continue
        e = abs(ar - CS_TARGET)
        errs.append(e)
        if bool(r.get("is_code_switched")) and e > 20 and len(cs_examples) < 8:
            cs_examples.append({"task": r["task"], "cs_ar_ratio": ar, "abs_err": round(e, 1),
                                "is_code_switched": True, "text": r["text"]})
    n = len(errs)
    cs_block = {
        "n": n,
        "within_10_pct": round(100 * sum(e <= 10 for e in errs) / n, 1) if n else None,
        "within_15_pct": round(100 * sum(e <= 15 for e in errs) / n, 1) if n else None,
        "within_20_pct": round(100 * sum(e <= 20 for e in errs) / n, 1) if n else None,
        "mean_abs_err": round(statistics.mean(errs), 2) if errs else None,
        "median_abs_err": round(statistics.median(errs), 2) if errs else None,
    }
    result["cs_ratio"] = cs_block

    # ---------- WRITE JSON ----------
    (DIR / "task_aware_error_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- WRITE CSV (per-sample with derived fields) ----------
    with open(DIR / "task_aware_error_analysis.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["task", "target_label", "predicted_or_relevant_or_passed", "task_correct",
                    "fluency", "naturalness", "high_quality_flu&nat>=8", "is_code_switched",
                    "cs_ar_ratio", "cs_ratio_abs_err_vs70", "ner_entity_counts", "ner_flags", "text"])
        for r in rows:
            ar = num(r.get("cs_ar_ratio"))
            pv = r.get("predicted") if r["task"] == "sentiment" else (
                r.get("topic_relevant") if r["task"] == "topic" else r.get("ner_passed"))
            hq = "yes" if (num(r.get("fluency")) or 0) >= 8 and (num(r.get("naturalness")) or 0) >= 8 else "no"
            flags = ""
            counts = ""
            if r["task"] == "ner":
                counts = json.dumps(r.get("entity_counts", {}), ensure_ascii=False)
                total = r.get("total_entities", 0) or 0
                fl = []
                if total == 0: fl.append("no_entities")
                if 0 < total < mn: fl.append("too_few")
                if total > mx: fl.append("too_many")
                for t in must:
                    if (r.get("entity_counts", {}) or {}).get(t, 0) == 0: fl.append(f"missing_{t}")
                flags = ";".join(fl)
            w.writerow([r["task"], r.get("target_label"), pv, r.get("task_correct"),
                        r.get("fluency"), r.get("naturalness"), hq, r.get("is_code_switched"),
                        ar, (round(abs(ar - CS_TARGET), 1) if ar is not None else ""),
                        counts, flags, r["text"]])

    # ---------- WRITE MD ----------
    L = ["# Test 1 — Error Analysis\n",
         f"Input: `{DETAILS.name}` · sentiment={len(sent)}, ner={len(ner)}, total={len(rows)}\n",
         "Goal: locate failures (prompt vs model vs evaluator artifact). No prompts changed.\n",
         "## 1. Sentiment errors\n",
         f"Per-label accuracy: " + ", ".join(
             f"{t}={sentiment_block['per_label_accuracy_pct'][t]}%" for t in labels) + "\n",
         "Confusion matrix (rows = target, cols = blind predicted):\n",
         "| target \\ predicted | " + " | ".join(cols) + " |",
         "|" + "---|" * (len(cols) + 1)]
    for t in labels:
        L.append(f"| {t} | " + " | ".join(str(confusion[t][c]) for c in cols) + " |")
    L.append("\n**Incorrect sentiment examples:**")
    for e in sent_errors[:6]:
        L.append(f"- target=`{e['target']}` predicted=`{e['predicted']}` (flu {e['fluency']}/nat {e['naturalness']}): {short(e['text'])}")

    L += ["\n## 2. NER errors\n", f"{ner_block['note']}. Failed {ner_failed}/{len(ner)}.\n",
          "Failure categories (a case can have several):\n",
          "| category | count |", "|---|---|"]
    for k, v in cats.items():
        L.append(f"| {k} | {v} |")
    L.append("\n**Failed NER examples:**")
    for e in ner_fail_examples[:6]:
        L.append(f"- counts={e['entity_counts']} total={e['total_entities']} flags={e['flags']}: {short(e['text'])}")

    L += ["\n## 3. Quality vs task mismatch\n",
          f"- mean fluency: correct={quality_block['mean_fluency_correct']} vs wrong={quality_block['mean_fluency_wrong']}",
          f"- mean naturalness: correct={quality_block['mean_naturalness_correct']} vs wrong={quality_block['mean_naturalness_wrong']}",
          f"- **fluency>=8 AND naturalness>=8 but task WRONG: {quality_block['high_quality_but_wrong_count']} cases "
          f"({quality_block['high_quality_but_wrong_pct_of_wrong']}% of all wrong)** — quality scores do not separate task success.\n",
          "**High-quality-but-wrong examples:**"]
    for e in quality_block["examples"][:6]:
        L.append(f"- {e['task']} target=`{e['target_label']}` flu {e['fluency']}/nat {e['naturalness']}: {short(e['text'])}")

    L += ["\n## 4. CS ratio control (target = 70% Arabic)\n",
          f"- within ±10: {cs_block['within_10_pct']}% · within ±15: {cs_block['within_15_pct']}% · within ±20: {cs_block['within_20_pct']}%",
          f"- mean abs error: {cs_block['mean_abs_err']} · median: {cs_block['median_abs_err']}\n",
          "**High CS-validity but poor ratio control (|err|>20):**"]
    for e in cs_examples[:6]:
        L.append(f"- {e['task']} ar_ratio={e['cs_ar_ratio']}% (err {e['abs_err']}): {short(e['text'])}")

    L += ["\n## Where do the failures come from? (reading)\n",
          "- **Sentiment**: check the confusion matrix — if errors cluster on one label (e.g. neutral<->positive), "
          "it is likely subjectivity / evaluator boundary, not pure generation failure.\n",
          "- **NER**: if failures are dominated by `missing_PER`/`missing_ORG` or `too_few`, the model is "
          "under-producing required entities (prompt/model), not an evaluator artifact.\n",
          "- **Quality mismatch**: high-quality-but-wrong cases prove quality scoring is blind to task success "
          "(motivates task-aware validation).\n",
          "- **CS ratio**: low within-±10 shows the model cannot self-regulate the requested proportion "
          "(model limitation), motivating deterministic CS-ratio control.\n"]
    (DIR / "task_aware_error_analysis.md").write_text("\n".join(L), encoding="utf-8")

    # console summary
    print("Sentiment per-label acc:", sentiment_block["per_label_accuracy_pct"])
    print("NER failure categories:", cats, f"(failed {ner_failed}/{len(ner)})")
    print(f"Quality: high-quality-but-wrong = {quality_block['high_quality_but_wrong_count']} "
          f"({quality_block['high_quality_but_wrong_pct_of_wrong']}% of wrong); "
          f"flu correct {quality_block['mean_fluency_correct']} vs wrong {quality_block['mean_fluency_wrong']}")
    print(f"CS ratio within +/-10/15/20: {cs_block['within_10_pct']}/{cs_block['within_15_pct']}/{cs_block['within_20_pct']}%")
    print(f"\nwrote task_aware_error_analysis.md/.csv/.json -> {DIR}")


if __name__ == "__main__":
    main()
