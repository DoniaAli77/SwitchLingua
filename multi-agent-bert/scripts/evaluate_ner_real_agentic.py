"""scripts/evaluate_ner_real_agentic.py

The payoff experiment: run the AGENTIC NER pipeline on the REAL Sabty
Arabic-English CS corpus and compare against the XLM-R baseline.

    XLM-R primary  ->  router  ->  (confident? keep primary)
                                ->  (unsure? LLM specialist -> consensus)

Everything runs in the corpus's TYPE label space {O, PERS, LOC, ORG, MISC}:
- XLM-R is normalised to type level (PER->PERS); it has NO MISC -> scores 0 on it.
- The LLM specialist CAN produce MISC (dynamic labels) -> fills the gap XLM-R can't.

Configs:
  1. XLMR_BASELINE   — XLM-R alone (primary_only). No LLM, no cost.
  2. AGENTIC_REFLECT — XLM-R primary -> router -> LLM reflects/corrects the draft.
  3. AGENTIC_TAGGER  — XLM-R primary -> router -> LLM tags & votes with the model.

NER-only: touches nothing in the sentiment/topic path.

Usage
-----
    python scripts/evaluate_ner_real_agentic.py --limit 150 --env_file ../Modified_Version/.env
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.ner_conll_loader import ENTITY_TYPES, TYPE_LABELS, load_conll, tag_to_type, to_type_dataset
from src.evaluation.ner_evaluator import NEREvaluator
from src.llm.mock_client import MockLLMClient
from src.llm.openai_client import OpenAIClient
from src.models.transformer_ner_tagger import TransformerNERTagger
from src.pipeline.ner_agentic import agentic_ner_task_config, build_agentic_ner_orchestrator

TEST = Path(__file__).resolve().parent.parent / "data" / "NER" / "Test_AR-EN_NER.txt"
OUT = Path(__file__).resolve().parent.parent / "experiments" / "outputs" / "multi_agent_bert" / "experiment_ner_real"

_LABEL_DESC = {
    "PERS": "a person's name (first, last, or full), Arabic or English",
    "LOC": "a location, city, country, or place",
    "ORG": "an organization, company, institution, or team",
    "MISC": "a named entity that is NOT a person/location/organization "
            "(events, nationalities, products, works, titles, competitions, etc.)",
}


def _load_env(env_file):
    for c in ([env_file] if env_file else [".env", "../.env"]):
        if not c:
            continue
        p = Path(c)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent / c
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return str(p)
    return None


def _type_macro(report):
    by = {m.label: m for m in report.per_tag}
    fs = [by[t].f1 for t in ENTITY_TYPES if t in by]
    return sum(fs) / len(fs) if fs else 0.0


def _run(name, orch, data, mode, threshold, agents_desc):
    tc = agentic_ner_task_config(TYPE_LABELS, threshold=threshold,
                                 pipeline_mode=mode, label_descriptions=_LABEL_DESC)
    rep = NEREvaluator(task_config=tc, orchestrator=orch, run_id=name).evaluate(data)
    by = {m.label: m for m in rep.per_tag}
    print(f"\n=== {name} ===  agents: {agents_desc}")
    print(f"  token_acc={rep.token_accuracy:.3f}  macro_f1(4 types)={_type_macro(rep):.3f}")
    for t in TYPE_LABELS:
        if t in by:
            m = by[t]
            print(f"    {t:<6} P={m.precision:.2f} R={m.recall:.2f} F1={m.f1:.2f} (n={m.support})")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Davlan/xlm-roberta-base-ner-hrl")
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--env_file", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    sents = to_type_dataset(load_conll(TEST))
    if args.limit:
        sents = sents[:args.limit]
    print(f"Loaded {len(sents)} REAL test sentences (type labels {TYPE_LABELS}).")

    print(f"Loading XLM-R ({args.model_dir or args.model}) with PER->PERS normalisation ...")
    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir or args.model, device=args.device,
        tag_normalizer=tag_to_type)

    reps = {}
    # 1) Baseline — XLM-R only (primary_only; LLM never called).
    base_orch = build_agentic_ner_orchestrator(
        tagger, MockLLMClient(mode="label_echo", allowed_labels=TYPE_LABELS),
        specialist="reflector")
    reps["baseline"] = _run("real_xlmr_baseline", base_orch, sents, "primary_only",
                            args.threshold, "XLM-R primary only")

    env_used = _load_env(args.env_file)
    if os.environ.get("OPENAI_API_KEY"):
        if env_used:
            print(f"\n[env loaded from {env_used}]")

        # 2) Agentic reflection — LLM reviews & corrects the draft.
        c_ref = OpenAIClient(model=args.llm_model, max_tokens=1500)
        ref_orch = build_agentic_ner_orchestrator(
            tagger, c_ref, specialist="reflector", model_weight=0.0, llm_weight=1.0)
        reps["reflect"] = _run("real_agentic_reflect", ref_orch, sents, "full_agentic",
                               args.threshold,
                               "XLM-R primary + router + LLM(gpt-4o-mini) REFLECTOR + consensus")

        # 3) Agentic voting — LLM tags & votes with the model.
        c_tag = OpenAIClient(model=args.llm_model, max_tokens=1500)
        tag_orch = build_agentic_ner_orchestrator(
            tagger, c_tag, specialist="tagger", model_weight=1.0, llm_weight=1.0)
        reps["tagger"] = _run("real_agentic_tagger", tag_orch, sents, "full_agentic",
                              args.threshold,
                              "XLM-R primary + router + LLM(gpt-4o-mini) TAGGER + consensus")

        try:
            ur, ut = c_ref.usage_summary(), c_tag.usage_summary()
            print(f"\nLLM calls — reflect: {ur['calls']} (esc.), tagger: {ut['calls']} (esc.)"
                  f"  cost≈${ur['est_cost_usd']+ut['est_cost_usd']:.4f}")
        except Exception:
            pass
    else:
        print("\n[!] OPENAI_API_KEY not found — ran baseline only. Pass --env_file <path>.")

    print("\n" + "=" * 46)
    print(f"{'CONFIG':<20}{'tok_acc':>9}{'macroF1':>9}{'MISC_F1':>9}")
    print("-" * 46)
    for k, rep in reps.items():
        misc = next((m.f1 for m in rep.per_tag if m.label == "MISC"), 0.0)
        print(f"{k:<20}{rep.token_accuracy:>9.3f}{_type_macro(rep):>9.3f}{misc:>9.2f}")
    print("=" * 46)
    print(f"\nSaved runs under: {OUT}")


if __name__ == "__main__":
    main()
