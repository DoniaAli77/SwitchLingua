"""
run_ner_constraint_difficulty.py — Explanatory NER difficulty analysis (does NOT change Test 1).
=================================================================================================
Question: is the low NER compliance (40%, English-only) because the model generally fails NER, or
because our config requires English-script PER + ORG in EVERY sentence (a hard multi-constraint)?

We generate small fresh samples under three constraint variants (same English-only policy,
gpt-4o-mini, refiner OFF) and score with the SAME final English-only NER judge:

  easy_ORG     : min 1, max 2, must_include [ORG]
  medium_PER   : min 1, max 2, must_include [PER]
  hard_PER_ORG : min 2, max 3, must_include [PER, ORG]   (= the current Test 1 config)

Policy held constant: target_entities_script=english, allow_code_switched_context=true,
allow_code_switched_entities=false.

Per variant (20 sentences): task_correct_pct, missing_PER, missing_ORG, count_valid_pct,
CS_validity_pct, fluency/naturalness mean.

Outputs (task_aware_eval/): ner_constraint_difficulty_summary.csv / .report.md
This is explanatory; it does NOT replace the final Test 1 NER number (40%).

Usage:
  python experiments/switchlingua/run_ner_constraint_difficulty.py --per-variant 20
"""
import argparse, csv, pathlib, statistics, sys

# Reuse run_task_aware_eval's setup: it activates the Modified core, imports node_engine as `ne`,
# creates `_llm`, and defines the final English-only `ner_ok`.
import run_task_aware_eval as t1
ne = t1.ne
ner_ok = t1.ner_ok
cs_stats = t1.compute_true_cs_stats
import utils  # noqa: E402  (Modified core, on sys.path via t1)

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = pathlib.Path(__file__).parent / "config_validation_100.yaml"
OUT = ROOT / "experiments" / "outputs" / "switchlingua" / "task_aware_eval"
OUT.mkdir(parents=True, exist_ok=True)

CS_TYPES = ["Intrasentential", "Intersentential", "Extra-sentential / Tag switching"]
BASE_POLICY = {
    "entity_types": ["PER", "ORG", "LOC"],
    "target_entities_script": ["english"],
    "allow_code_switched_context": [True],
    "allow_code_switched_entities": [False],
}
VARIANTS = {
    "easy_ORG":     {"min_entities": [1], "max_entities": [2], "must_include_types": ["ORG"]},
    "medium_PER":   {"min_entities": [1], "max_entities": [2], "must_include_types": ["PER"]},
    "hard_PER_ORG": {"min_entities": [2], "max_entities": [3], "must_include_types": ["PER", "ORG"]},
}


def build_scenarios(variant_ner):
    cfg = utils.load_config(str(CONFIG))["pre_execute"]
    cfg["task"] = ["ner"]
    cfg["shared"]["cs_type"] = CS_TYPES
    cfg["shared"]["topic"] = ["tech", "finance"]
    cfg["ner"] = {**BASE_POLICY, **variant_ner}
    return utils.generate_scenarios(cfg)


def gen_and_score(scenarios, n_target):
    rows = []
    for sc in scenarios:
        if len(rows) >= n_target:
            break
        state = dict(sc)
        state["news_article"] = ""; state["news_hash"] = set()
        state["news_dict"] = {}; state["mcp_result"] = ""
        try:
            gen = ne.RunDataGenerationAgent(state)
        except Exception as e:
            print(f"   gen failed: {type(e).__name__}: {str(e)[:60]}")
            continue
        sents = gen.get("data_generation_result", []) or []
        if not sents:
            continue
        state["data_generation_result"] = sents
        try:
            flu = ne.RunFluencyAgent(state).get("fluency_results_per_instances", []) or []
            nat = ne.RunNaturalnessAgent(state).get("naturalness_results_per_instances", []) or []
        except Exception as e:
            print(f"   scoring failed: {type(e).__name__}: {str(e)[:60]}")
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
            rows.append({
                "text": txt.strip(),
                "is_cs": bool(st.get("is_code_switched")),
                "fluency": (flu[j].get("fluency_score") if j < len(flu) and isinstance(flu[j], dict) else None),
                "naturalness": (nat[j].get("naturalness_score") if j < len(nat) and isinstance(nat[j], dict) else None),
                "passed": nres.get("passed"),
                "PER": counts.get("PER", 0), "ORG": counts.get("ORG", 0), "LOC": counts.get("LOC", 0),
                "total": nres.get("total_entities"),
                "count_valid": nres.get("count_valid"),
            })
    return rows


def pct(flags):
    flags = [f for f in flags if isinstance(f, bool)]
    return round(100 * sum(flags) / len(flags), 1) if flags else None


def mean(vals):
    vals = [float(v) for v in vals if v is not None]
    return round(statistics.mean(vals), 2) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-variant", type=int, default=20)
    a = ap.parse_args()

    summary = []
    for name, vner in VARIANTS.items():
        must = "+".join(vner["must_include_types"])
        print(f"[{name}] must={must} count={vner['min_entities'][0]}-{vner['max_entities'][0]} — generating {a.per_variant} ...")
        scen = build_scenarios(vner)
        rows = gen_and_score(scen, a.per_variant)
        n = len(rows)
        summary.append({
            "variant": name,
            "min_entities": vner["min_entities"][0], "max_entities": vner["max_entities"][0],
            "must_include": must, "n": n,
            "task_correct_pct": pct([r["passed"] for r in rows]),
            "missing_PER": sum(1 for r in rows if r["PER"] == 0),
            "missing_ORG": sum(1 for r in rows if r["ORG"] == 0),
            "count_valid_pct": pct([r["count_valid"] for r in rows]),
            "cs_validity_pct": pct([r["is_cs"] for r in rows]),
            "fluency_mean": mean([r["fluency"] for r in rows]),
            "naturalness_mean": mean([r["naturalness"] for r in rows]),
        })
        s = summary[-1]
        print(f"   -> task_correct={s['task_correct_pct']}%  missing_PER={s['missing_PER']}/{n}  "
              f"missing_ORG={s['missing_ORG']}/{n}  count_valid={s['count_valid_pct']}%  CS={s['cs_validity_pct']}%")

    with open(OUT / "ner_constraint_difficulty_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    L = ["# NER Constraint-Difficulty Analysis (explanatory — does NOT change the final Test 1 NER score)\n",
         "Question: is low NER compliance a general model failure, or an artifact of requiring English-script "
         "**PER + ORG in every sentence**? Same English-only policy, gpt-4o-mini, refiner OFF, final English-only judge.\n",
         f"Sample: {a.per_variant} NER sentences per variant.\n",
         "| variant | must-include | count | n | task-correct % | missing_PER | missing_ORG | count-valid % | CS-valid % | fluency | naturalness |",
         "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for s in summary:
        L.append(f"| {s['variant']} | {s['must_include']} | {s['min_entities']}-{s['max_entities']} | {s['n']} | "
                 f"{s['task_correct_pct']} | {s['missing_PER']} | {s['missing_ORG']} | {s['count_valid_pct']} | "
                 f"{s['cs_validity_pct']} | {s['fluency_mean']} | {s['naturalness_mean']} |")
    L += ["\n## How to read\n",
          "- If **easy_ORG / medium_PER score much higher than hard_PER_ORG**, the low Test 1 NER (40%) is largely "
          "driven by the **strict PER+ORG-in-every-sentence config**, not a general NER failure.\n",
          "- Compare **missing_PER vs missing_ORG**: if missing_PER stays high even in medium_PER, the model "
          "specifically struggles to produce **English-script PERSON** names (it tends to write person references "
          "in Arabic inside the Arabic-matrix sentence).\n",
          "- The final Test 1 NER number (40%, hard_PER_ORG English-only) is unchanged; this only explains *why*.\n"]
    (OUT / "ner_constraint_difficulty_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote ner_constraint_difficulty_summary.csv / .report.md -> {OUT}")


if __name__ == "__main__":
    main()
