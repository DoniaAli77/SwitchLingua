"""
run_ner_per_ab.py — Did generalizing the NER prompt hurt PER? Controlled same-session A/B.
==========================================================================================
Compares, on the SAME PER_ORG config (must PER+ORG, 2-3, English-only judge, refiner OFF):
  per_block : the OLD forceful PER-specific section-11 wording (reconstructed in the harness)
  generic   : the CURRENT core prompt (dynamic {ner_entity_guidance})
Both arms run in one session so the comparison isolates the wording (not run-to-run variance).

Outputs: experiments/outputs/switchlingua/task_aware_eval/ner_per_ab_summary.csv
"""
import argparse, csv, math, pathlib, statistics

import run_task_aware_eval as t1
ne = t1.ne
ner_ok = t1.ner_ok
cs_stats = t1.compute_true_cs_stats
import importlib
cov = importlib.import_module("run_ner_tag_coverage")
from prompt import DATA_GENERATION_NER_PROMPT
from node_models import GenerationResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

OUT = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "outputs" / "switchlingua" / "task_aware_eval"

OLD_PER_BLOCK = (
    "            11. ADDITIONAL STRICT PERSON-ENTITY REQUIREMENT (English-script):\n"
    "            - Every instance MUST contain at least one English-script FULL PERSON name written in Latin letters whenever PER is required by must_include_types.\n"
    '              Use natural names such as "Ahmed Ali", "Sarah Hassan", "Omar Khaled", "Mona Ibrahim", or "Elon Musk".\n'
    "            - Do NOT write required PERSON or ORGANIZATION entities in Arabic script; only the surrounding context may be Arabic.\n"
    '            - If ORG is required, also include at least one English-script organization name such as "Google", "Microsoft", "Apple", or "Cairo University".\n'
    "            - SELF-CHECK each instance before finalizing:\n"
    "              (1) at least one English-script full PERSON name appears when PER is required,\n"
    "              (2) at least one English-script ORGANIZATION appears when ORG is required,\n"
    "              (3) the total number of named entities is within min_entities..max_entities.\n"
)


def build_per_block_prompt():
    """Current core template with section 11 swapped for the OLD forceful PER block."""
    tmpl = DATA_GENERATION_NER_PROMPT.messages[0].prompt.template
    i = tmpl.index("11. REQUIRED ENTITY GUIDANCE")
    # back up to the start of that line (keep leading indentation)
    line_start = tmpl.rfind("\n", 0, i) + 1
    j = tmpl.index("Now, generate code-switched NER data")
    new_tmpl = tmpl[:line_start] + OLD_PER_BLOCK + "\n            " + tmpl[j:]
    return ChatPromptTemplate.from_messages([("assistant", new_tmpl)])


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    c = (p + z*z/(2*n)) / (1 + z*z/n)
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return (round(100*(c-h), 1), round(100*(c+h), 1))


def gen(chain, state, inject_guidance, retries=4):
    s = dict(state)
    s["news_article"] = ""; s["news_hash"] = set(); s["news_dict"] = {}; s["mcp_result"] = ""
    if inject_guidance:
        s["ner_entity_guidance"] = ne.build_ner_entity_guidance(s.get("task_constraints", {}))
    resp = chain.invoke(s)
    n = retries
    while (not isinstance(resp, dict) or not resp.get("instances")) and n > 0:
        resp = chain.invoke(s); n -= 1
    return (resp.get("instances", []) if isinstance(resp, dict) else []), s


def run_arm(name, chain, inject, scenarios, n_target):
    rows = []
    for sc in scenarios:
        if len(rows) >= n_target:
            break
        try:
            sents, s = gen(chain, sc, inject)
        except Exception as e:
            print(f"   [{name}] gen failed: {type(e).__name__}: {str(e)[:60]}"); continue
        if not sents:
            continue
        s["data_generation_result"] = sents
        try:
            flu = ne.RunFluencyAgent(s).get("fluency_results_per_instances", []) or []
            nat = ne.RunNaturalnessAgent(s).get("naturalness_results_per_instances", []) or []
        except Exception:
            flu, nat = [], []
        cons = s.get("task_constraints", {})
        for k, txt in enumerate(sents):
            if len(rows) >= n_target:
                break
            if not isinstance(txt, str) or not txt.strip():
                continue
            jr = ner_ok(txt, cons); counts = jr.get("entity_counts", {}) or {}
            rows.append({"passed": jr.get("passed"), "PER": counts.get("PER", 0), "ORG": counts.get("ORG", 0),
                         "count_valid": jr.get("count_valid"), "is_cs": bool(cs_stats(txt).get("is_code_switched")),
                         "fluency": (flu[k].get("fluency_score") if k < len(flu) and isinstance(flu[k], dict) else None),
                         "naturalness": (nat[k].get("naturalness_score") if k < len(nat) and isinstance(nat[k], dict) else None)})
        print(f"   [{name}] {len(rows)}/{n_target}")
    return rows


def summarize(name, rows):
    n = len(rows)
    pc = [r for r in rows if isinstance(r["passed"], bool)]
    k = sum(1 for r in pc if r["passed"])
    def mean(xs):
        xs = [float(x) for x in xs if x is not None]
        return round(statistics.mean(xs), 2) if xs else None
    return {"arm": name, "n": n, "task_correct_pct": round(100*k/len(pc), 1) if pc else None,
            "task_correct_ci95": f"{wilson(k,len(pc))[0]}-{wilson(k,len(pc))[1]}",
            "missing_PER": sum(1 for r in rows if r["PER"] == 0),
            "missing_ORG": sum(1 for r in rows if r["ORG"] == 0),
            "count_valid_pct": round(100*sum(1 for r in rows if r["count_valid"])/n, 1) if n else None,
            "cs_validity_pct": round(100*sum(1 for r in rows if r["is_cs"])/n, 1) if n else None,
            "fluency_mean": mean([r["fluency"] for r in rows]), "naturalness_mean": mean([r["naturalness"] for r in rows])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-arm", type=int, default=30); a = ap.parse_args()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, base_url=ne.API_BASE, api_key=ne.API_KEY
                     ).with_structured_output(GenerationResponse)
    arms = [("per_block", build_per_block_prompt() | llm, False),
            ("generic", DATA_GENERATION_NER_PROMPT | llm, True)]
    summaries = []
    for name, chain, inject in arms:
        print(f"[{name}] generating {a.per_arm} PER_ORG ...")
        rows = run_arm(name, chain, inject, cov.build_scenarios(["PER", "ORG"], 2, 3), a.per_arm)
        summaries.append(summarize(name, rows))
        s = summaries[-1]
        print(f"   => task_correct={s['task_correct_pct']}% CI{s['task_correct_ci95']} missing_PER={s['missing_PER']}/{s['n']}")
    with open(OUT / "ner_per_ab_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys())); w.writeheader(); w.writerows(summaries)
    pb = next(s for s in summaries if s["arm"] == "per_block")
    gn = next(s for s in summaries if s["arm"] == "generic")
    print(f"\nPER_OR B A/B: per_block {pb['task_correct_pct']}% (missing_PER {pb['missing_PER']}/{pb['n']}) "
          f"vs generic {gn['task_correct_pct']}% (missing_PER {gn['missing_PER']}/{gn['n']})")
    print("READ: if generic << per_block AND missing_PER much higher in generic -> generalizing weakened PER "
          "(strengthen generic wording). If similar -> the 25% smoke was variance.")


if __name__ == "__main__":
    main()
