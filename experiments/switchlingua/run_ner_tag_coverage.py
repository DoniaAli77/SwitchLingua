"""
run_ner_tag_coverage.py — NER tag coverage / generalization (does NOT change Test 1).
=====================================================================================
After the PER-focused prompt repair, test whether NER works for entity types beyond PER/ORG
(adds PRODUCT, EVENT). Same final policy: target_entities_script=english,
allow_code_switched_context=true, allow_code_switched_entities=false, refiner OFF, gpt-4o-mini,
same English-only ner_ok judge.

Core entity types offered to generator + judge every run: PER, ORG, LOC, PRODUCT, EVENT.

Group 1 (single-tag capability, 20 each):
  single_PER / single_ORG / single_LOC / single_PRODUCT / single_EVENT  (min 1, max 2)
Group 2 (pairwise constraints, 18 each):
  PER_ORG / PER_LOC / ORG_PRODUCT / EVENT_LOC / PER_EVENT  (min 2, max 3)

Per variant: task_correct_pct, missing count per required type, disallowed-type count,
count_valid_pct, CS_validity_pct, fluency/naturalness mean (+ Wilson CI on task_correct).

Outputs: experiments/outputs/switchlingua/task_aware_eval/ner_tag_coverage_summary.csv / report.md
"""
import argparse, csv, math, pathlib, statistics

import run_task_aware_eval as t1           # activates Modified core; provides ne, ner_ok, compute_true_cs_stats
ne = t1.ne
ner_ok = t1.ner_ok
cs_stats = t1.compute_true_cs_stats
import utils

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = pathlib.Path(__file__).parent / "config_validation_100.yaml"
OUT = ROOT / "experiments" / "outputs" / "switchlingua" / "task_aware_eval"
OUT.mkdir(parents=True, exist_ok=True)

CORE_TYPES = ["PER", "ORG", "LOC", "PRODUCT", "EVENT"]
CS_TYPES = ["Intrasentential", "Intersentential", "Extra-sentential / Tag switching"]
POLICY = {"target_entities_script": ["english"], "allow_code_switched_context": [True],
          "allow_code_switched_entities": [False]}

GROUP1 = {  # (must_include, min, max)
    "single_PER": (["PER"], 1, 2), "single_ORG": (["ORG"], 1, 2), "single_LOC": (["LOC"], 1, 2),
    "single_PRODUCT": (["PRODUCT"], 1, 2), "single_EVENT": (["EVENT"], 1, 2),
}
GROUP2 = {
    "PER_ORG": (["PER", "ORG"], 2, 3), "PER_LOC": (["PER", "LOC"], 2, 3),
    "ORG_PRODUCT": (["ORG", "PRODUCT"], 2, 3), "EVENT_LOC": (["EVENT", "LOC"], 2, 3),
    "PER_EVENT": (["PER", "EVENT"], 2, 3),
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    c = (p + z*z/(2*n)) / (1 + z*z/n)
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return (round(100*(c-h), 1), round(100*(c+h), 1))


def build_scenarios(must, mn, mx):
    cfg = utils.load_config(str(CONFIG))["pre_execute"]
    cfg["task"] = ["ner"]
    cfg["shared"]["cs_type"] = CS_TYPES
    cfg["shared"]["topic"] = ["tech", "finance"]
    cfg["ner"] = {"entity_types": CORE_TYPES, "min_entities": [mn], "max_entities": [mx],
                  "must_include_types": must, **POLICY}
    return utils.generate_scenarios(cfg)


def generate(state, retries=4):
    resp = ne.RunDataGenerationAgent(state)
    sents = resp.get("data_generation_result", []) if isinstance(resp, dict) else []
    n = retries
    while not sents and n > 0:
        resp = ne.RunDataGenerationAgent(state)
        sents = resp.get("data_generation_result", []) if isinstance(resp, dict) else []
        n -= 1
    return sents


def run_variant(name, must, mn, mx, n_target):
    rows = []
    for sc in build_scenarios(must, mn, mx):
        if len(rows) >= n_target:
            break
        state = dict(sc)
        state["news_article"] = ""; state["news_hash"] = set(); state["news_dict"] = {}; state["mcp_result"] = ""
        try:
            sents = generate(state)
        except Exception as e:
            print(f"   [{name}] gen failed: {type(e).__name__}"); continue
        if not sents:
            continue
        state["data_generation_result"] = sents
        try:
            flu = ne.RunFluencyAgent(state).get("fluency_results_per_instances", []) or []
            nat = ne.RunNaturalnessAgent(state).get("naturalness_results_per_instances", []) or []
        except Exception:
            flu, nat = [], []
        cons = state.get("task_constraints", {})
        for j, txt in enumerate(sents):
            if len(rows) >= n_target:
                break
            if not isinstance(txt, str) or not txt.strip():
                continue
            st = cs_stats(txt)
            jr = ner_ok(txt, cons)
            counts = jr.get("entity_counts", {}) or {}
            rows.append({
                "is_cs": bool(st.get("is_code_switched")),
                "fluency": (flu[j].get("fluency_score") if j < len(flu) and isinstance(flu[j], dict) else None),
                "naturalness": (nat[j].get("naturalness_score") if j < len(nat) and isinstance(nat[j], dict) else None),
                "passed": jr.get("passed"), "counts": counts,
                "count_valid": jr.get("count_valid"),
                "disallowed": jr.get("disallowed_types", []) or [],
            })
    return rows


def pctf(flags):
    flags = [f for f in flags if isinstance(f, bool)]
    return (round(100*sum(flags)/len(flags), 1), sum(flags), len(flags)) if flags else (None, 0, 0)


def meanf(vals):
    vals = [float(v) for v in vals if v is not None]
    return round(statistics.mean(vals), 2) if vals else None


def summarize(group, name, must, mn, mx, rows):
    n = len(rows)
    tc, k, tot = pctf([r["passed"] for r in rows])
    cv, _, _ = pctf([r["count_valid"] for r in rows])
    cs, _, _ = pctf([r["is_cs"] for r in rows])
    missing = {t: sum(1 for r in rows if r["counts"].get(t, 0) == 0) for t in must}
    disallowed_ct = sum(1 for r in rows if r["disallowed"])
    return {
        "group": group, "variant": name, "must_include": "+".join(must),
        "min": mn, "max": mx, "n": n,
        "task_correct_pct": tc, "task_correct_ci95": f"{wilson(k,tot)[0]}-{wilson(k,tot)[1]}",
        "missing_required": ";".join(f"{t}:{missing[t]}/{n}" for t in must),
        "disallowed_type_count": disallowed_ct,
        "count_valid_pct": cv, "cs_validity_pct": cs,
        "fluency_mean": meanf([r["fluency"] for r in rows]),
        "naturalness_mean": meanf([r["naturalness"] for r in rows]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-n", type=int, default=20)
    ap.add_argument("--pair-n", type=int, default=18)
    a = ap.parse_args()

    summaries = []
    for name, (must, mn, mx) in GROUP1.items():
        print(f"[G1 {name}] generating {a.single_n} ...")
        rows = run_variant(name, must, mn, mx, a.single_n)
        summaries.append(summarize("single", name, must, mn, mx, rows))
        s = summaries[-1]
        print(f"   -> task_correct={s['task_correct_pct']}% missing={s['missing_required']} "
              f"disallowed={s['disallowed_type_count']} CS={s['cs_validity_pct']}%")
    for name, (must, mn, mx) in GROUP2.items():
        print(f"[G2 {name}] generating {a.pair_n} ...")
        rows = run_variant(name, must, mn, mx, a.pair_n)
        summaries.append(summarize("pairwise", name, must, mn, mx, rows))
        s = summaries[-1]
        print(f"   -> task_correct={s['task_correct_pct']}% missing={s['missing_required']} "
              f"disallowed={s['disallowed_type_count']} CS={s['cs_validity_pct']}%")

    with open(OUT / "ner_tag_coverage_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys())); w.writeheader(); w.writerows(summaries)

    L = ["# NER Tag Coverage / Generalization (explanatory — does NOT change Test 1)\n",
         "After the PER-focused prompt repair: does NER work for entity types beyond PER/ORG (adds PRODUCT, EVENT)?\n",
         "Policy: English-only target entities, allow_code_switched_context, refiner OFF, gpt-4o-mini, English-only judge.\n",
         "## Group 1 — single-tag capability\n",
         "| variant | must | count | n | task-correct % (95% CI) | missing (req types) | disallowed | count-valid % | CS % | flu | nat |",
         "|---|---|---|--:|--:|---|--:|--:|--:|--:|--:|"]
    for s in [x for x in summaries if x["group"] == "single"]:
        L.append(f"| {s['variant']} | {s['must_include']} | {s['min']}-{s['max']} | {s['n']} | {s['task_correct_pct']} ({s['task_correct_ci95']}) | "
                 f"{s['missing_required']} | {s['disallowed_type_count']} | {s['count_valid_pct']} | {s['cs_validity_pct']} | {s['fluency_mean']} | {s['naturalness_mean']} |")
    L += ["\n## Group 2 — pairwise constraints\n",
          "| variant | must | count | n | task-correct % (95% CI) | missing (req types) | disallowed | count-valid % | CS % | flu | nat |",
          "|---|---|---|--:|--:|---|--:|--:|--:|--:|--:|"]
    for s in [x for x in summaries if x["group"] == "pairwise"]:
        L.append(f"| {s['variant']} | {s['must_include']} | {s['min']}-{s['max']} | {s['n']} | {s['task_correct_pct']} ({s['task_correct_ci95']}) | "
                 f"{s['missing_required']} | {s['disallowed_type_count']} | {s['count_valid_pct']} | {s['cs_validity_pct']} | {s['fluency_mean']} | {s['naturalness_mean']} |")
    L += ["\n## How to read\n",
          "- **Which tags generalize?** High task-correct + low missing for a type = the model produces that "
          "English-script entity type reliably. Likely order: ORG/LOC easy, PER moderate (post-repair), "
          "PRODUCT/EVENT unknown — this experiment measures them.\n",
          "- **missing (req types)** shows per-type absence; a type with high 'missing' is the bottleneck in that variant.\n",
          "- **disallowed** = entities the judge labeled outside the allowed set (should be ~0).\n",
          "- Small n (18–20) → wide CIs and run-to-run variance; read this as coverage breadth, not precise rates. "
          "Does NOT change the main Test 1 NER number.\n"]
    (OUT / "ner_tag_coverage_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote ner_tag_coverage_summary.csv / report.md -> {OUT}")


if __name__ == "__main__":
    main()
