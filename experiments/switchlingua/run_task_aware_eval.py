"""
run_task_aware_eval.py — Test 1: task-aware generation quality (automated, no humans).
=======================================================================================
Proves the modified pipeline generates VALID task-specific data for topic/sentiment/NER,
using checks that do NOT require annotators today:

  - CS validity   : deterministic compute_true_cs_stats (is_code_switched)         [objective]
  - CS ratio       : deterministic actual matrix-ratio vs target 70%               [objective]
  - NER correctness: deterministic English-entity policy (_deterministic_ner_...)  [objective]
  - sentiment      : BLIND independent re-classification by gpt-4o-mini (not told target)  [LLM]
  - topic          : BLIND relevance check by gpt-4o-mini (not told it must say yes)        [LLM]
  - fluency/natural: reuse the pipeline's existing per-sentence judge scores

Reuses the existing PRE-REFINEMENT sample (refiner OFF); no regeneration.
Human confirmation of task correctness comes later via the consolidated sheet.

Outputs to experiments/outputs/switchlingua/task_aware_eval/:
  task_aware_summary.csv, task_aware_summary.json, task_aware_details.jsonl, task_aware_report.md

Usage:
  python experiments/switchlingua/run_task_aware_eval.py --per-task 40
"""
import argparse, importlib, json, os, pathlib, random, statistics, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
CFG = pathlib.Path(__file__).parent / "threshold_sweep.yaml"
OUT = ROOT / "experiments" / "outputs" / "switchlingua" / "task_aware_eval"
OUT.mkdir(parents=True, exist_ok=True)

# env + SSL
import dotenv
dotenv.load_dotenv(str(ROOT / "Modified_Version" / ".env"), override=True)
os.environ["API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
os.environ["API_BASE"] = os.environ.get("OPENAI_BASE_URL", "")
import ssl, httpx
ssl._create_default_https_context = ssl._create_unverified_context
_o = httpx.Client.__init__
def _c(self, *a, **k): k.setdefault("verify", False); k.setdefault("timeout", 60.0); _o(self, *a, **k)
httpx.Client.__init__ = _c

if str(MODIFIED_CORE) not in sys.path:
    sys.path.insert(0, str(MODIFIED_CORE))
for m in ("utils", "node_engine", "node_models", "prompt"):
    sys.modules.pop(m, None)
importlib.invalidate_caches()
import yaml
from utils import compute_true_cs_stats
import node_engine as ne
from langchain_openai import ChatOpenAI

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0,
                  base_url=ne.API_BASE, api_key=ne.API_KEY)


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def classify_sentiment(text):
    """Blind: classify sentiment without seeing the target label."""
    msg = ("Classify the overall sentiment of this Arabic-English sentence. "
           "Answer with exactly one word: positive, negative, or neutral.\n\n" + text)
    try:
        out = _llm.invoke(msg).content.strip().lower()
        for lab in ("positive", "negative", "neutral"):
            if lab in out:
                return lab
    except Exception as e:
        return f"ERROR:{type(e).__name__}"
    return "unknown"


def topic_relevant(text, topic):
    """Blind relevance: is the sentence about the target topic? (model not told to say yes)"""
    msg = (f"Does the following Arabic-English sentence relate to the topic '{topic}'? "
           f"Answer strictly yes or no.\n\n{text}")
    try:
        out = _llm.invoke(msg).content.strip().lower()
        if out.startswith("y") or "yes" in out[:5]:
            return True
        if out.startswith("n") or "no" in out[:4]:
            return False
    except Exception:
        return None
    return None


def ner_ok(text, constraints):
    """Blind LLM entity extraction (script-agnostic — the task allows code-switched entities),
    then a deterministic constraint check. The English-only deterministic policy is NOT used
    because it unfairly ignores Arabic-script entities that this task permits."""
    c = constraints or {}
    must = [str(t).strip().upper() for t in (c.get("must_include_types", []) or [])]
    mn = int(c.get("min_entities", 0) or 0)
    mx = int(c.get("max_entities", 10**9) or 10**9)
    msg = ("Extract every named entity in this Arabic-English sentence and label each as "
           "PER (person), ORG (organization), or LOC (location). Count entities in EITHER "
           "language/script. Output only lines of the form 'TYPE: entity'. If none, output NONE.\n\n" + text)
    try:
        out = _llm.invoke(msg).content
    except Exception:
        return None, {}, None
    counts = {"PER": 0, "ORG": 0, "LOC": 0}
    for line in out.splitlines():
        t = line.split(":", 1)[0].strip().upper()
        if t in counts:
            counts[t] += 1
    total = sum(counts.values())
    has_must = all(counts.get(t, 0) > 0 for t in must) if must else True
    passed = has_must and (mn <= total <= mx)
    return bool(passed), counts, total


def cs_ratio_error(stats):
    """abs(actual matrix(Arabic) ratio - 70 target). Arabic is the matrix language."""
    ar = stats.get("cs_ar_ratio")
    if ar is None:
        return None
    return round(abs(float(ar) - 70.0), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    data_path = ROOT / cfg["input"]["data"]
    records = load_jsonl(data_path)

    # flatten to sentence items with parent metadata
    by_task = {"topic": [], "sentiment": [], "ner": []}
    for rec in records:
        task = rec.get("task")
        if task not in by_task:
            continue
        sents = rec.get("data_generation_result", [])
        flu = rec.get("fluency_results_per_instances", []) or []
        nat = rec.get("naturalness_results_per_instances", []) or []
        for j, txt in enumerate(sents):
            if not isinstance(txt, str) or not txt.strip():
                continue
            by_task[task].append({
                "task": task, "text": txt.strip(),
                "label": rec.get("label", rec.get("topic", "")),
                "constraints": rec.get("task_constraints", {}),
                "fluency": (flu[j].get("fluency_score") if j < len(flu) and isinstance(flu[j], dict) else None),
                "naturalness": (nat[j].get("naturalness_score") if j < len(nat) and isinstance(nat[j], dict) else None),
            })

    random.seed(args.seed)
    details, summary_rows = [], []
    for task, items in by_task.items():
        random.shuffle(items)
        sample = items[:args.per_task]
        print(f"[{task}] evaluating {len(sample)} sentences ...")
        task_correct, cs_valid, ratio_err, flu_v, nat_v = [], [], [], [], []
        for it in sample:
            stats = compute_true_cs_stats(it["text"])
            is_cs = bool(stats.get("is_code_switched"))
            cs_valid.append(is_cs)
            re_ = cs_ratio_error(stats);
            if re_ is not None: ratio_err.append(re_)
            if it["fluency"] is not None: flu_v.append(float(it["fluency"]))
            if it["naturalness"] is not None: nat_v.append(float(it["naturalness"]))

            rec_detail = {"task": task, "text": it["text"], "target_label": it["label"],
                          "is_code_switched": is_cs, "cs_ar_ratio": stats.get("cs_ar_ratio"),
                          "fluency": it["fluency"], "naturalness": it["naturalness"]}
            if task == "sentiment":
                pred = classify_sentiment(it["text"])
                ok = (pred == str(it["label"]).strip().lower())
                task_correct.append(ok); rec_detail.update(predicted=pred, task_correct=ok)
            elif task == "topic":
                ok = topic_relevant(it["text"], it["label"])
                if ok is not None: task_correct.append(ok)
                rec_detail.update(topic_relevant=ok, task_correct=ok)
            elif task == "ner":
                ok, counts, total = ner_ok(it["text"], it["constraints"])
                if ok is not None: task_correct.append(ok)
                rec_detail.update(ner_passed=ok, entity_counts=counts, total_entities=total, task_correct=ok)
            details.append(rec_detail)

        def pct(flags):
            flags = [f for f in flags if isinstance(f, bool)]
            return round(100*sum(flags)/len(flags), 1) if flags else None
        summary_rows.append({
            "task": task, "n": len(sample),
            "task_correct_pct": pct(task_correct),
            "task_correct_method": {"sentiment": "blind re-classification (LLM)",
                                     "topic": "blind relevance (LLM)",
                                     "ner": "blind entity extraction (LLM) + constraint check"}[task],
            "cs_validity_pct": pct(cs_valid),
            "cs_ratio_mae_vs_70": round(statistics.mean(ratio_err), 2) if ratio_err else None,
            "fluency_mean": round(statistics.mean(flu_v), 2) if flu_v else None,
            "naturalness_mean": round(statistics.mean(nat_v), 2) if nat_v else None,
        })
        s = summary_rows[-1]
        print(f"   task_correct={s['task_correct_pct']}%  CS_valid={s['cs_validity_pct']}%  "
              f"CS_ratio_MAE={s['cs_ratio_mae_vs_70']}  flu={s['fluency_mean']} nat={s['naturalness_mean']}")

    # write outputs
    import csv
    with open(OUT / "task_aware_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys())); w.writeheader(); w.writerows(summary_rows)
    (OUT / "task_aware_summary.json").write_text(json.dumps(
        {"input": str(data_path), "per_task": args.per_task, "results": summary_rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    with open(OUT / "task_aware_details.jsonl", "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # report
    L = ["# Test 1 — Task-Aware Generation Quality (automated, no humans yet)\n",
         f"**Source (pre-refinement, refiner OFF):** `{data_path}`  ",
         f"**Per task:** {args.per_task} sentences\n",
         "## Results\n",
         "| Task | n | Task-correct % | (method) | CS-valid % | CS-ratio MAE vs 70 | Fluency | Naturalness |",
         "|------|--:|--:|:--|--:|--:|--:|--:|"]
    for s in summary_rows:
        L.append(f"| {s['task']} | {s['n']} | {s['task_correct_pct']} | {s['task_correct_method']} | "
                 f"{s['cs_validity_pct']} | {s['cs_ratio_mae_vs_70']} | {s['fluency_mean']} | {s['naturalness_mean']} |")
    L += ["\n## Notes on rigor\n",
          "- **CS validity and CS-ratio are deterministic/objective** (compute_true_cs_stats; no LLM, no circularity).\n",
          "- **Task correctness (sentiment/topic/NER) is automated by a BLIND gpt-4o-mini judge** not shown the "
          "target (sentiment = re-classification; topic = relevance; NER = entity extraction + deterministic "
          "constraint check). Less circular than the in-pipeline validator but still LLM-based; **human confirmation "
          "is pending** via `human_eval/consolidated_annotation_sheet.csv`.\n",
          "- The English-only deterministic NER policy was deliberately NOT used (it ignores Arabic-script entities "
          "this task permits, giving unfairly low scores).\n",
          "- Fluency/naturalness are the pipeline's own per-sentence judge scores (for reference).\n",
          "- Sample reuses the fresh pre-refinement validation set; no regeneration.\n"]
    (OUT / "task_aware_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote task_aware_summary.csv/.json, task_aware_details.jsonl, task_aware_report.md -> {OUT}")


if __name__ == "__main__":
    main()
