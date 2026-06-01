"""
run_task_validator_necessity.py — Test 2: TaskValidatorAgent necessity / effectiveness.
=======================================================================================
Replays two acceptance policies over the EXISTING Test 1 outputs. It does NOT regenerate
data; it DOES run the real TaskValidatorAgent on the already-generated sentences to obtain
its independent pass/fail verdict (Policy B needs the validator's own decision).

Three distinct signals (kept separate — no oracle/circularity):
  - REFERENCE task-correctness = Test 1 `task_correct` (blind LLM judge; swap for human via --labels)
  - VALIDATOR verdict          = the real TaskValidatorAgent's passed/failed (run here)
  - QUALITY pass               = pipeline per-sentence weighted score >= threshold (from validation_raw)

Policies:
  A. quality_only:                accept if quality_pass                       (ignore validator)
  B. quality_plus_task_validator: accept if quality_pass AND validator_passed

Detection framing vs the REFERENCE (want: accept task-correct, reject task-wrong):
  TP accept&correct | FP accept&wrong (false accept) | FN reject&correct | TN reject&wrong
  FP_rate = FP/(FP+TN) = share of task-WRONG that get accepted
  FN_rate = FN/(FN+TP) = share of task-CORRECT that get rejected

Validator verdicts are cached in task_validator/validator_verdicts.jsonl (use --refresh to rerun).

Outputs (experiments/outputs/switchlingua/task_validator/):
  task_validator_necessity_summary.csv / .json / report.md

Usage:
  python experiments/switchlingua/run_task_validator_necessity.py            # run validator + analyze
  python experiments/switchlingua/run_task_validator_necessity.py --limit 6  # smoke test
  python experiments/switchlingua/run_task_validator_necessity.py --analyze-only
  python experiments/switchlingua/run_task_validator_necessity.py --labels human.csv --analyze-only
"""
import argparse, csv, importlib, json, os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
CFG = pathlib.Path(__file__).parent / "threshold_sweep.yaml"
DETAILS = ROOT / "experiments" / "outputs" / "switchlingua" / "task_aware_eval" / "task_aware_details.jsonl"
RAW = ROOT / "experiments" / "outputs" / "switchlingua" / "per_sentence" / "validation_raw" / "Arabic.jsonl"
OUT = ROOT / "experiments" / "outputs" / "switchlingua" / "task_validator"
OUT.mkdir(parents=True, exist_ok=True)
VERDICTS = OUT / "validator_verdicts.jsonl"


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_maps():
    """text -> (weighted_quality_score, source_record) from validation_raw."""
    qmap, recmap = {}, {}
    for r in load_jsonl(RAW):
        scores = r.get("sentence_scores", []) or []
        for j, txt in enumerate(r.get("data_generation_result", []) or []):
            if isinstance(txt, str):
                key = txt.strip()
                if j < len(scores):
                    qmap[key] = float(scores[j])
                recmap[key] = r
    return qmap, recmap


# ---------------- Phase 1: run the real TaskValidatorAgent (API) ----------------
def run_validator(details, recmap, limit=None):
    # activate Modified core + env/SSL
    import dotenv
    dotenv.load_dotenv(str(ROOT / "Modified_Version" / ".env"), override=True)
    os.environ["API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
    os.environ["API_BASE"] = os.environ.get("OPENAI_BASE_URL", "")
    import ssl, httpx
    ssl._create_default_https_context = ssl._create_unverified_context
    _o = httpx.Client.__init__
    httpx.Client.__init__ = lambda self, *a, **k: (k.setdefault("verify", False), k.setdefault("timeout", 60.0), _o(self, *a, **k))[-1]
    if str(MODIFIED_CORE) not in sys.path:
        sys.path.insert(0, str(MODIFIED_CORE))
    for m in ("utils", "node_engine", "node_models", "prompt", "mcp_tools", "agents", "run_french"):
        sys.modules.pop(m, None)
    importlib.invalidate_caches()
    import node_engine as ne
    ne.MODEL = "gpt-4o-mini"

    rows = details if limit is None else details[:limit]
    verdicts = []
    for i, d in enumerate(rows, 1):
        txt = d["text"].strip()
        rec = recmap.get(txt)
        if rec is None:
            verdicts.append({"text": txt, "task": d["task"], "validator_passed": None, "note": "no_source_record"})
            continue
        state = dict(rec)
        state["data_generation_result"] = [txt]
        try:
            res = ne.RunTaskValidatorAgent(state)
            agg = res.get("task_validation_result", {}) if isinstance(res, dict) else {}
            passed = bool(agg.get("passed", False))
            verdicts.append({"text": txt, "task": d["task"], "validator_passed": passed,
                             "predicted_label": agg.get("predicted_label")})
        except Exception as e:
            verdicts.append({"text": txt, "task": d["task"], "validator_passed": None,
                             "note": f"{type(e).__name__}: {str(e)[:80]}"})
        if i % 10 == 0 or i == len(rows):
            print(f"  validator {i}/{len(rows)}")
    with open(VERDICTS, "w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    return verdicts


# ---------------- Phase 2: metrics ----------------
def metrics(rows, accept_key):
    tp = fp = fn = tn = 0
    for r in rows:
        acc, cor = r[accept_key], r["correct"]
        if acc and cor: tp += 1
        elif acc and not cor: fp += 1
        elif not acc and cor: fn += 1
        else: tn += 1
    accepted = tp + fp
    return {"n_eval": len(rows), "accepted": accepted,
            "accepted_pct": round(100*accepted/len(rows), 1) if rows else None,
            "task_correct_among_accepted_pct": round(100*tp/accepted, 1) if accepted else None,
            "task_wrong_accepted": fp,
            "fp_rate_pct": round(100*fp/(fp+tn), 1) if (fp+tn) else None,
            "fn_rate_pct": round(100*fn/(fn+tp), 1) if (fn+tp) else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def analyze(labels_csv=None):
    cfg = __import__("yaml").safe_load(open(CFG, encoding="utf-8"))
    threshold = float(cfg["acceptance_threshold"])
    qmap, _ = build_maps()
    details = load_jsonl(DETAILS)
    verdicts = {v["text"]: v for v in load_jsonl(VERDICTS)} if VERDICTS.exists() else {}

    human, ref_name = {}, "Test 1 blind LLM judge"
    if labels_csv and pathlib.Path(labels_csv).exists():
        for r in csv.DictReader(open(labels_csv, encoding="utf-8-sig")):
            human[(r.get("text") or "").strip()] = (r.get("task_correct") or "").strip().lower() in {"yes","y","1","true"}
        ref_name = f"human labels ({labels_csv})"

    rows, no_q, no_v = [], 0, 0
    for d in details:
        txt = d["text"].strip()
        q = qmap.get(txt)
        if q is None:
            no_q += 1; continue
        v = verdicts.get(txt, {})
        vp = v.get("validator_passed")
        if vp is None:
            no_v += 1; continue
        correct = human.get(txt, d.get("task_correct"))
        if correct is None:
            continue
        qpass = q >= threshold
        rows.append({"task": d["task"], "text": txt, "quality_score": round(q, 3),
                     "quality_pass": qpass, "validator_passed": bool(vp), "correct": bool(correct),
                     "accept_A": qpass, "accept_B": qpass and bool(vp),
                     "fluency": d.get("fluency"), "naturalness": d.get("naturalness")})

    def subset(s): return rows if s == "overall" else [r for r in rows if r["task"] == s]
    summary = {"reference": ref_name, "validator": "real TaskValidatorAgent (gpt-4o-mini)",
               "quality_threshold": threshold, "n_evaluated": len(rows),
               "unmatched_quality": no_q, "missing_validator_verdict": no_v, "by_scope": {}}
    csv_rows = []
    for scope in ["overall", "topic", "sentiment", "ner"]:
        rs = subset(scope)
        if not rs: continue
        a, b = metrics(rs, "accept_A"), metrics(rs, "accept_B")
        summary["by_scope"][scope] = {"quality_only": a, "quality_plus_validator": b}
        for pol, m in (("quality_only", a), ("quality_plus_validator", b)):
            csv_rows.append({"scope": scope, "policy": pol, **{k: m[k] for k in
                ["n_eval","accepted","accepted_pct","task_correct_among_accepted_pct",
                 "task_wrong_accepted","fp_rate_pct","fn_rate_pct"]}})

    # validator's own accuracy vs reference (as a task-correctness detector)
    vtp=vfp=vfn=vtn=0
    for r in rows:
        vp, cor = r["validator_passed"], r["correct"]
        if vp and cor: vtp+=1
        elif vp and not cor: vfp+=1
        elif not vp and cor: vfn+=1
        else: vtn+=1
    summary["validator_vs_reference"] = {
        "agreement_pct": round(100*(vtp+vtn)/len(rows),1) if rows else None,
        "validator_precision_pct": round(100*vtp/(vtp+vfp),1) if (vtp+vfp) else None,  # of validator-pass, how many truly correct
        "validator_recall_pct": round(100*vtp/(vtp+vfn),1) if (vtp+vfn) else None,     # of correct, how many validator passed
        "tp": vtp, "fp": vfp, "fn": vfn, "tn": vtn}

    # high-quality but task-wrong that quality_only accepts (and whether validator caught them)
    hq_wrong = [r for r in rows if r["quality_pass"] and not r["correct"]
                and (r["fluency"] or 0) >= 8 and (r["naturalness"] or 0) >= 8]
    caught = sum(1 for r in hq_wrong if not r["validator_passed"])
    summary["high_quality_but_wrong"] = {
        "count": len(hq_wrong), "caught_by_validator": caught,
        "examples": [{"task": r["task"], "quality_score": r["quality_score"], "fluency": r["fluency"],
                      "naturalness": r["naturalness"], "validator_passed": r["validator_passed"],
                      "text": r["text"]} for r in hq_wrong[:10]]}

    # write CSV + JSON
    with open(OUT / "task_validator_necessity_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["scope","policy","n_eval","accepted","accepted_pct",
            "task_correct_among_accepted_pct","task_wrong_accepted","fp_rate_pct","fn_rate_pct"])
        w.writeheader(); w.writerows(csv_rows)
    (OUT / "task_validator_necessity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # report
    ov = summary["by_scope"]["overall"]; A, B = ov["quality_only"], ov["quality_plus_validator"]
    vv = summary["validator_vs_reference"]
    L = ["# Test 2 — TaskValidatorAgent Necessity & Effectiveness\n",
         f"Reference task-correctness: **{ref_name}** · Validator: **real TaskValidatorAgent (gpt-4o-mini)** · "
         f"quality threshold (weighted score) = **{threshold}** · evaluated **{len(rows)}** sentences.\n",
         "Policy **A: quality_only** (accept if quality>=threshold) vs **B: quality + TaskValidator** "
         "(accept if quality passes AND the real validator passes).\n",
         "## Overall (vs reference)\n",
         "| metric | A: quality_only | B: quality + validator |", "|---|---|---|",
         f"| accepted | {A['accepted']} ({A['accepted_pct']}%) | {B['accepted']} ({B['accepted_pct']}%) |",
         f"| task-correct among accepted | {A['task_correct_among_accepted_pct']}% | {B['task_correct_among_accepted_pct']}% |",
         f"| **task-WRONG accepted (false accepts)** | **{A['task_wrong_accepted']}** | **{B['task_wrong_accepted']}** |",
         f"| false-accept rate (of wrong) | {A['fp_rate_pct']}% | {B['fp_rate_pct']}% |",
         f"| false-reject rate (of correct) | {A['fn_rate_pct']}% | {B['fn_rate_pct']}% |",
         "\n## The validator as a task-correctness detector (vs reference)\n",
         f"- agreement {vv['agreement_pct']}% · precision {vv['validator_precision_pct']}% · recall {vv['validator_recall_pct']}%  "
         f"(TP {vv['tp']}, FP {vv['fp']}, FN {vv['fn']}, TN {vv['tn']})\n",
         "## Per task\n",
         "| task | policy | accepted | task-correct among accepted | task-wrong accepted | false-accept % |",
         "|---|---|---|---|---|---|"]
    for scope in ["topic","sentiment","ner"]:
        if scope not in summary["by_scope"]: continue
        for pol in ("quality_only","quality_plus_validator"):
            m = summary["by_scope"][scope][pol]
            L.append(f"| {scope} | {pol} | {m['accepted']} | {m['task_correct_among_accepted_pct']}% | "
                     f"{m['task_wrong_accepted']} | {m['fp_rate_pct']}% |")
    L += [f"\n## High-quality but task-wrong (quality_only accepts these)\n",
          f"**{summary['high_quality_but_wrong']['count']}** sentences pass quality AND have fluency>=8 & "
          f"naturalness>=8 but are task-WRONG; the real validator caught **{caught}** of them. Examples:\n"]
    for e in summary["high_quality_but_wrong"]["examples"][:8]:
        L.append(f"- {e['task']} (q={e['quality_score']}, flu {e['fluency']}/nat {e['naturalness']}, "
                 f"validator_passed={e['validator_passed']}): {e['text'][:100]}")
    L += ["\n## Interpretation\n",
          f"- Quality-only admits **{A['task_wrong_accepted']}** task-wrong sentences "
          f"(false-accept {A['fp_rate_pct']}% of all wrong) — they look fluent/natural, so quality cannot separate them.\n",
          f"- Adding the real TaskValidator cuts task-wrong accepted to **{B['task_wrong_accepted']}** "
          f"(false-accept {B['fp_rate_pct']}%), raising precision from {A['task_correct_among_accepted_pct']}% to "
          f"{B['task_correct_among_accepted_pct']}% — at a false-reject cost of {B['fn_rate_pct']}%.\n",
          f"- The validator is imperfect (precision {vv['validator_precision_pct']}%, recall {vv['validator_recall_pct']}% "
          "vs the reference), so the gain is real but not total. Reference can be swapped for human labels via --labels.\n"]
    (OUT / "task_validator_necessity_report.md").write_text("\n".join(L), encoding="utf-8")

    print(f"reference={ref_name} threshold={threshold} evaluated={len(rows)} (no_q={no_q}, no_validator={no_v})")
    print(f"A quality_only: accepted {A['accepted']}, task-wrong accepted {A['task_wrong_accepted']}, "
          f"precision {A['task_correct_among_accepted_pct']}%, false-accept {A['fp_rate_pct']}%")
    print(f"B quality+validator: accepted {B['accepted']}, task-wrong accepted {B['task_wrong_accepted']}, "
          f"precision {B['task_correct_among_accepted_pct']}%, false-accept {B['fp_rate_pct']}%, false-reject {B['fn_rate_pct']}%")
    print(f"validator vs reference: precision {vv['validator_precision_pct']}% recall {vv['validator_recall_pct']}% "
          f"agreement {vv['agreement_pct']}%")
    print(f"high-quality-but-wrong: {summary['high_quality_but_wrong']['count']} (validator caught {caught})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only run validator on first N")
    ap.add_argument("--analyze-only", action="store_true", help="skip validator run; use cached verdicts")
    ap.add_argument("--refresh", action="store_true", help="force re-run the validator")
    ap.add_argument("--labels", default=None, help="human-label CSV (text, task_correct) to replace reference")
    a = ap.parse_args()

    details = load_jsonl(DETAILS)
    if not a.analyze_only and (a.refresh or not VERDICTS.exists() or a.limit):
        _, recmap = build_maps()
        print(f"Running real TaskValidatorAgent on {a.limit or len(details)} sentences ...")
        run_validator(details, recmap, limit=a.limit)
    if a.limit:
        print("(smoke run complete — verdicts cached; run without --limit for full, then --analyze-only)")
        return
    analyze(labels_csv=a.labels)


if __name__ == "__main__":
    main()
