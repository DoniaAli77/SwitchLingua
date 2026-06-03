"""Final NER smoke through the REAL core pipeline (CodeSwitchingAgent graph) with the REAL
TaskValidator. Validator ON, refiner OFF, English-only policy. Reports the requested metrics."""
import asyncio, pathlib, statistics

import run_task_aware_eval as t1            # activates Modified core
ne = t1.ne; ner_ok = t1.ner_ok; cs_stats = t1.compute_true_cs_stats
import utils
import run_french as rf

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = pathlib.Path(__file__).parent / "config_validation_100.yaml"
TMP = ROOT / "experiments" / "outputs" / "switchlingua" / "task_aware_eval" / "_realpipe_tmp"
TMP.mkdir(parents=True, exist_ok=True)

ne.MODEL = "gpt-4o-mini"
ne.OUTPUT_DIR = str(TMP)
rf.ENABLE_TASK_VALIDATOR = True      # REAL TaskValidator ON  (set before building the graph)
rf.MAX_SENTENCE_REFINES = 0          # refiner OFF
ne.MAX_SENTENCE_REFINES = 0


def build_scenarios(n_scen=6):
    cfg = utils.load_config(str(CONFIG))["pre_execute"]
    cfg["task"] = ["ner"]
    cfg["shared"]["cs_type"] = ["Intrasentential", "Intersentential", "Extra-sentential / Tag switching"]
    cfg["shared"]["topic"] = ["tech", "finance"]
    return utils.generate_scenarios(cfg)[:n_scen]


async def main():
    scen = build_scenarios(6)
    gen_count = parse_ok = task_correct = 0
    val_pass = val_total = 0
    miss = {"PER": 0, "ORG": 0}
    cs_ok = 0
    flu, nat = [], []
    errors = []
    judged = 0
    for i, sc in enumerate(scen):
        try:
            agent = rf.CodeSwitchingAgent(dict(sc))
            state = await agent.run() or {}
        except Exception as e:
            errors.append(f"scenario {i}: {type(e).__name__}: {str(e)[:80]}")
            continue
        sents = state.get("data_generation_result", []) or []
        per_val = state.get("task_validation_results_per_instances", []) or []
        fl = state.get("fluency_results_per_instances", []) or []
        na = state.get("naturalness_results_per_instances", []) or []
        cons = state.get("task_constraints", {})
        for j, txt in enumerate(sents):
            if not isinstance(txt, str) or not txt.strip():
                continue
            gen_count += 1
            jr = ner_ok(txt, cons)
            if jr.get("parse_error") is None:
                parse_ok += 1
            if jr.get("passed") is True:
                task_correct += 1
            judged += 1
            counts = jr.get("entity_counts", {}) or {}
            if counts.get("PER", 0) == 0: miss["PER"] += 1
            if counts.get("ORG", 0) == 0: miss["ORG"] += 1
            if bool(cs_stats(txt).get("is_code_switched")): cs_ok += 1
            if j < len(fl) and isinstance(fl[j], dict) and fl[j].get("fluency_score") is not None:
                flu.append(float(fl[j]["fluency_score"]))
            if j < len(na) and isinstance(na[j], dict) and na[j].get("naturalness_score") is not None:
                nat.append(float(na[j]["naturalness_score"]))
            if j < len(per_val) and isinstance(per_val[j], dict):
                val_total += 1
                if per_val[j].get("passed"): val_pass += 1
        print(f"  scenario {i+1}/{len(scen)} done ({len(sents)} sentences)")

    m = lambda xs: round(statistics.mean(xs), 2) if xs else None
    print("\n==== FINAL NER REAL-PIPELINE SMOKE (validator ON, refiner OFF, English-only) ====")
    print(f"  generated sentences:        {gen_count}")
    print(f"  parse success (judge JSON): {parse_ok}/{judged}")
    print(f"  task_correct (Eng-only judge): {task_correct}/{judged} = {round(100*task_correct/judged,1) if judged else None}%")
    print(f"  validator pass rate:        {val_pass}/{val_total} = {round(100*val_pass/val_total,1) if val_total else None}%")
    print(f"  missing types:              PER {miss['PER']}/{judged}, ORG {miss['ORG']}/{judged}")
    print(f"  CS validity:                {cs_ok}/{judged} = {round(100*cs_ok/judged,1) if judged else None}%")
    print(f"  fluency / naturalness mean: {m(flu)} / {m(nat)}")
    print(f"  errors:                     {errors if errors else 'none'}")


if __name__ == "__main__":
    asyncio.run(main())
