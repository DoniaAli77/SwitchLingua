"""scripts/evaluate_ner_sequential.py

Sequential multi-agent NER panel on REAL Sabty test data, matching the
literature's staged design (KDR-Agent / CROSSAGENTIE):

    primary (fine-tuned XLM-R)  ->  router  ->  (accept if confident)
                                            ->  escalate:
                                                  reflector  (review & correct draft)
                                                  debate     (resolve primary-vs-reflector clashes)
                                                  disambiguation (re-type entities)

Reports the score after EACH stage, so you can see each added agent's marginal
effect vs the primary-only baseline. Everything runs in the type label space
{O, PERS, LOC, ORG, MISC}. NER-only; sentiment/topic untouched.

Usage
-----
    python scripts/evaluate_ner_sequential.py --model_dir models/xlmr_sabty_ner \\
        --limit 100 --threshold 0.90 --env_file ../Modified_Version/.env
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

from src.agents.llm_ner_reflection_agent import LLMNERReflectionAgent
from src.agents.llm_ner_panel_agents import LLMNERDebateAgent, LLMNERDisambiguationAgent
from src.evaluation.evaluator import _per_class_metrics
from src.evaluation.ner_conll_loader import ENTITY_TYPES, TYPE_LABELS, load_conll, tag_to_type, to_type_dataset
from src.llm.openai_client import OpenAIClient
from src.models.transformer_ner_tagger import TransformerNERTagger
from src.pipeline.ner_agentic import agentic_ner_task_config
from src.state.schema import PipelineState, StateMetadata

TEST = Path(__file__).resolve().parent.parent / "data" / "NER" / "Test_AR-EN_NER.txt"
_DESC = {
    "PERS": "a person's name (Arabic or English)",
    "LOC": "a location/city/country/place", "ORG": "an organization/company/team",
    "MISC": "a named entity that is NOT person/location/organization (events, "
            "nationalities, products, competitions, titles)",
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


def _macro(gold, pred):
    per = {m.label: m for m in _per_class_metrics(gold, pred, TYPE_LABELS)}
    fs = [per[t].f1 for t in ENTITY_TYPES if t in per]
    misc = per.get("MISC")
    return (sum(fs) / len(fs) if fs else 0.0), (misc.f1 if misc else 0.0)


def _report(name, agents, gold, pred):
    """Detailed per-type report (matches the earlier before/after format)."""
    per = {m.label: m for m in _per_class_metrics(gold, pred, TYPE_LABELS)}
    tok = sum(g == p for g, p in zip(gold, pred)) / max(1, len(gold))
    fs = [per[t].f1 for t in ENTITY_TYPES if t in per]
    macro = sum(fs) / len(fs) if fs else 0.0
    print(f"\n=== {name} ===  agents: {agents}")
    print(f"  token_acc={tok:.3f}  macro_f1(4 types)={macro:.3f}")
    for t in TYPE_LABELS:
        if t in per:
            m = per[t]
            print(f"    {t:<6} P={m.precision:.2f} R={m.recall:.2f} F1={m.f1:.2f} (n={m.support})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Davlan/xlm-roberta-base-ner-hrl")
    ap.add_argument("--model_dir", default="models/xlmr_sabty_ner")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--env_file", default=None)
    args = ap.parse_args()

    sents = to_type_dataset(load_conll(TEST))[:args.limit] if args.limit else to_type_dataset(load_conll(TEST))
    print(f"Loaded {len(sents)} REAL test sentences.")
    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir or args.model, device=args.device, tag_normalizer=tag_to_type)

    if not _load_env(args.env_file) or not os.environ.get("OPENAI_API_KEY"):
        print("[!] OPENAI_API_KEY not found — pass --env_file <path>."); return
    client = OpenAIClient(model=args.llm_model, max_tokens=1500)
    reflector = LLMNERReflectionAgent(client, output_slot="contextual")
    debate = LLMNERDebateAgent(client, source_a="model", source_b="contextual", output_slot="contextual")
    disambig = LLMNERDisambiguationAgent(client, source_slot="contextual", output_slot="contextual")

    # Collect gold + predictions after each stage.
    gold_all = []
    stages = {"primary": [], "reflect": [], "debate": [], "disambig": []}
    escalated = 0
    for s in sents:
        gold_all.extend(s["tags"])
        st = PipelineState(metadata=StateMetadata(sample_id=s["id"]), input_text=s["text"],
                           task_config=agentic_ner_task_config(TYPE_LABELS, threshold=args.threshold,
                                                               label_descriptions=_DESC),
                           extras={"tokens": s["tokens"]})
        tagger.run(st)
        prim = [t.tag for t in st.ner_model_output.sequence_output.tags]
        confs = [t.confidence for t in st.ner_model_output.sequence_output.tags]
        stages["primary"].extend(prim)

        if confs and min(confs) >= args.threshold:      # router accepts -> no agents
            for k in ("reflect", "debate", "disambig"):
                stages[k].extend(prim)
            continue
        escalated += 1
        reflector.run(st)
        stages["reflect"].extend([t.tag for t in st.contextual_output.sequence_output.tags])
        debate.run(st)
        stages["debate"].extend([t.tag for t in st.contextual_output.sequence_output.tags])
        disambig.run(st)
        stages["disambig"].extend([t.tag for t in st.contextual_output.sequence_output.tags])

    # Detailed before/after reports (like last time).
    _report("BEFORE  (primary only)", "fine-tuned XLM-R", gold_all, stages["primary"])
    _report("AFTER   (full panel)",
            "XLM-R + router + reflector + debate + disambiguation",
            gold_all, stages["disambig"])

    print(f"\nEscalated {escalated}/{len(sents)} sentences. LLM calls: {client.usage_summary()['calls']}"
          f"  cost≈${client.usage_summary()['est_cost_usd']:.4f}")
    print("\n" + "=" * 48)
    print(f"{'STAGE (cumulative)':<26}{'macroF1':>10}{'MISC':>10}")
    print("-" * 48)
    for k in ("primary", "reflect", "debate", "disambig"):
        m, misc = _macro(gold_all, stages[k])
        label = {"primary": "primary (before)", "reflect": "+ reflector",
                 "debate": "+ debate", "disambig": "+ disambiguation (full)"}[k]
        print(f"{label:<26}{m:>10.3f}{misc:>10.2f}")
    print("=" * 48)


if __name__ == "__main__":
    main()
