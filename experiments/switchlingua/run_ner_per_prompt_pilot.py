"""
run_ner_per_prompt_pilot.py — PER-focused NER prompt repair: rigorous before/after pilot.
=========================================================================================
Tests whether emphasizing English-script PERSON names in the NER generation prompt improves
PER compliance — WITHOUT modifying the core prompt (the variant is defined here and only
*promoted* to core if it wins).

Two arms (hard_PER_ORG config: must PER+ORG, 2-3 entities; English-only target policy):
  current     : the live core DATA_GENERATION_NER_PROMPT (unchanged)
  per_focused : core prompt + an appended English-script-PERSON requirement + self-check

Both arms: gpt-4o-mini, temperature 0.7, refiner OFF, scored by the SAME English-only ner_ok judge.
N >= 50 per arm (because same-config NER varies ~40-60% run-to-run, so 30 would be underpowered).
Reports task_correct (+ Wilson 95% CI), missing_PER/ORG, count_valid, CS-validity (+CI),
CS-ratio MAE vs 70 (side-effect check), fluency/naturalness.

Outputs: experiments/outputs/switchlingua/task_aware_eval/ner_per_prompt_repair/

Usage:
  python experiments/switchlingua/run_ner_per_prompt_pilot.py --per-arm 50
"""
import argparse, csv, math, pathlib, statistics

import run_task_aware_eval as t1          # activates Modified core; provides ne, ner_ok, compute_true_cs_stats
ne = t1.ne
ner_ok = t1.ner_ok
cs_stats = t1.compute_true_cs_stats
import utils                               # Modified core (on path via t1)
from prompt import DATA_GENERATION_NER_PROMPT
from node_models import GenerationResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = pathlib.Path(__file__).parent / "config_validation_100.yaml"
OUT = ROOT / "experiments" / "outputs" / "switchlingua" / "task_aware_eval" / "ner_per_prompt_repair"
OUT.mkdir(parents=True, exist_ok=True)

CS_TYPES = ["Intrasentential", "Intersentential", "Extra-sentential / Tag switching"]
HARD = {"min_entities": [2], "max_entities": [3], "must_include_types": ["PER", "ORG"]}
BASE_POLICY = {"entity_types": ["PER", "ORG", "LOC"], "target_entities_script": ["english"],
               "allow_code_switched_context": [True], "allow_code_switched_entities": [False]}

PER_EMPHASIS = ("assistant", """
ADDITIONAL STRICT PERSON-ENTITY REQUIREMENT (English-script):
- Every instance MUST contain at least one English-script FULL PERSON name written in Latin letters,
  even though the surrounding sentence is Arabic. Use natural names such as:
  "Ahmed Ali", "Sarah Hassan", "Omar Khaled", "Mona Ibrahim", "Elon Musk".
- Do NOT write the required person/organization entities in Arabic script; only the surrounding
  context may be Arabic.
- If ORG is required, also include at least one English-script organization name (e.g. "Google", "Microsoft").
- SELF-CHECK each instance before finalizing and regenerate if any of these fail:
  (1) at least one English-script full PERSON name is present,
  (2) at least one English-script ORGANIZATION is present (when ORG is required),
  (3) the total number of named entities is within min_entities..max_entities.
""")
VARIANT_PROMPT = ChatPromptTemplate.from_messages(list(DATA_GENERATION_NER_PROMPT.messages) + [PER_EMPHASIS])


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    c = (p + z*z/(2*n)) / (1 + z*z/n)
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return (round(100*(c-h), 1), round(100*(c+h), 1))


def build_scenarios():
    cfg = utils.load_config(str(CONFIG))["pre_execute"]
    cfg["task"] = ["ner"]
    cfg["shared"]["cs_type"] = CS_TYPES
    cfg["shared"]["topic"] = ["tech", "finance", "health", "sports"]   # 4 topics x 3 cs = 12 NER scenarios
    cfg["ner"] = {**BASE_POLICY, **HARD}
    return utils.generate_scenarios(cfg)


def generate(chain, state, retries=4):
    resp = chain.invoke(state)
    n = retries
    while (not isinstance(resp, dict) or not resp.get("instances")) and n > 0:
        resp = chain.invoke(state); n -= 1
    return resp.get("instances", []) if isinstance(resp, dict) else []


def run_arm(name, chain, scenarios, n_target):
    rows = []
    for sc in scenarios:
        if len(rows) >= n_target:
            break
        state = dict(sc)
        state["news_article"] = ""; state["news_hash"] = set(); state["news_dict"] = {}; state["mcp_result"] = ""
        try:
            sents = generate(chain, state)
        except Exception as e:
            print(f"   [{name}] gen failed: {type(e).__name__}: {str(e)[:60]}"); continue
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
            nres = ner_ok(txt, cons)
            counts = nres.get("entity_counts", {}) or {}
            ar = st.get("cs_ar_ratio")
            rows.append({
                "arm": name, "text": txt.strip(), "is_cs": bool(st.get("is_code_switched")),
                "cs_ar_ratio": ar, "cs_ratio_abs_err": (abs(float(ar)-70.0) if ar is not None else None),
                "fluency": (flu[j].get("fluency_score") if j < len(flu) and isinstance(flu[j], dict) else None),
                "naturalness": (nat[j].get("naturalness_score") if j < len(nat) and isinstance(nat[j], dict) else None),
                "passed": nres.get("passed"), "PER": counts.get("PER", 0), "ORG": counts.get("ORG", 0),
                "total": nres.get("total_entities"), "count_valid": nres.get("count_valid"),
            })
        print(f"   [{name}] {len(rows)}/{n_target}")
    return rows


def pct(flags):
    flags = [f for f in flags if isinstance(f, bool)]
    return (round(100*sum(flags)/len(flags), 1), sum(flags), len(flags)) if flags else (None, 0, 0)


def mean(vals):
    vals = [float(v) for v in vals if v is not None]
    return round(statistics.mean(vals), 2) if vals else None


def summarize(name, rows):
    n = len(rows)
    tc, tck, tcn = pct([r["passed"] for r in rows])
    cv, _, _ = pct([r["count_valid"] for r in rows])
    cs, csk, csn = pct([r["is_cs"] for r in rows])
    return {
        "arm": name, "n": n,
        "task_correct_pct": tc, "task_correct_ci95": wilson(tck, tcn),
        "missing_PER": sum(1 for r in rows if r["PER"] == 0),
        "missing_ORG": sum(1 for r in rows if r["ORG"] == 0),
        "count_valid_pct": cv,
        "cs_validity_pct": cs, "cs_validity_ci95": wilson(csk, csn),
        "cs_ratio_mae_vs_70": mean([r["cs_ratio_abs_err"] for r in rows]),
        "fluency_mean": mean([r["fluency"] for r in rows]),
        "naturalness_mean": mean([r["naturalness"] for r in rows]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-arm", type=int, default=50)
    a = ap.parse_args()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, base_url=ne.API_BASE, api_key=ne.API_KEY
                     ).with_structured_output(GenerationResponse)
    arms = {"current": DATA_GENERATION_NER_PROMPT | llm, "per_focused": VARIANT_PROMPT | llm}

    summaries, all_rows = [], []
    for name, chain in arms.items():
        print(f"[{name}] generating {a.per_arm} hard_PER_ORG NER sentences ...")
        rows = run_arm(name, chain, build_scenarios(), a.per_arm)
        all_rows += rows
        summaries.append(summarize(name, rows))
        s = summaries[-1]
        print(f"   => task_correct={s['task_correct_pct']}% CI{s['task_correct_ci95']} "
              f"missing_PER={s['missing_PER']}/{s['n']} CS_ratio_MAE={s['cs_ratio_mae_vs_70']} CS={s['cs_validity_pct']}%")

    with open(OUT / "ner_per_prompt_pilot_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys())); w.writeheader(); w.writerows(summaries)
    with open(OUT / "ner_per_prompt_pilot_details.csv", "w", newline="", encoding="utf-8-sig") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)

    cur = next(s for s in summaries if s["arm"] == "current")
    pf = next(s for s in summaries if s["arm"] == "per_focused")
    L = ["# NER PER-focused Prompt Repair — before/after pilot\n",
         "Variant prompt tested in the harness only (core prompt NOT modified). hard_PER_ORG config, "
         "English-only judge, gpt-4o-mini @0.7, refiner OFF.\n",
         f"N per arm: {a.per_arm}. CIs are Wilson 95%.\n",
         "| arm | n | task-correct % (95% CI) | missing_PER | missing_ORG | count-valid % | CS-valid % | CS-ratio MAE vs70 | fluency | naturalness |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for s in summaries:
        L.append(f"| {s['arm']} | {s['n']} | {s['task_correct_pct']} ({s['task_correct_ci95'][0]}–{s['task_correct_ci95'][1]}) | "
                 f"{s['missing_PER']} | {s['missing_ORG']} | {s['count_valid_pct']} | {s['cs_validity_pct']} | "
                 f"{s['cs_ratio_mae_vs_70']} | {s['fluency_mean']} | {s['naturalness_mean']} |")
    L += ["\n## How to read\n",
          f"- **Real improvement?** Compare task-correct CIs: current {cur['task_correct_ci95']} vs "
          f"per_focused {pf['task_correct_ci95']}. If the CIs **overlap heavily**, the change is within "
          "run-to-run noise (recall same-config NER varies ~40–60%) — not a confirmed win.\n",
          "- **Did PER improve?** missing_PER should drop in per_focused.\n",
          "- **Side effects?** Watch CS-ratio MAE and naturalness — forcing English names can push text toward "
          "English (worse ratio) or read less naturally. A 'fix' that breaks the CS ratio is not a win.\n",
          "- Still LLM-judged (blind judge); a confirmed win needs human spot-check. Core prompt unchanged; "
          "promote the variant only if it clearly wins without side effects.\n"]
    (OUT / "ner_per_prompt_pilot_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote summary.csv / details.csv / report.md -> {OUT}")
    print(f"current: {cur['task_correct_pct']}% CI{cur['task_correct_ci95']} | per_focused: {pf['task_correct_pct']}% CI{pf['task_correct_ci95']}")


if __name__ == "__main__":
    main()
