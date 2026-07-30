"""scripts/evaluate_ner_span_compare.py

Isolate the alignment-drift effect: on the REAL test data, compare the LLM's OWN
tagging quality under two output formats, head-to-head:

  POSITIONAL : LLMNERAgent      — LLM emits one tag per token (drifts).
  SPAN       : LLMNERSpanAgent  — LLM lists {text,type}; deterministic aligner
                                  places them (no drift).

Reports, over the ESCALATED (hard) sentences, each method's macro-F1 vs gold,
plus the primary on the same sentences. If SPAN >> POSITIONAL, drift was a real
technical cause; if they're similar, the damage is genuine disagreement.

    python scripts/evaluate_ner_span_compare.py --model_dir models/xlmr_sabty_ner \\
        --limit 100 --env_file ../Modified_Version/.env
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

from src.agents.llm_ner_agent import LLMNERAgent
from src.agents.llm_ner_span_agent import LLMNERSpanAgent
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
        p = Path(c) if c else None
        if p and not p.is_absolute():
            p = Path(__file__).resolve().parent.parent / c
        if p and p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return str(p)
    return None


def _macro(gold, pred):
    per = {m.label: m for m in _per_class_metrics(gold, pred, TYPE_LABELS)}
    fs = [per[t].f1 for t in ENTITY_TYPES if t in per]
    return (sum(fs) / len(fs) if fs else 0.0), (per["MISC"].f1 if "MISC" in per else 0.0)


def _fresh(s, thr):
    return PipelineState(metadata=StateMetadata(sample_id=s["id"]), input_text=s["text"],
        task_config=agentic_ner_task_config(TYPE_LABELS, threshold=thr, label_descriptions=_DESC),
        extras={"tokens": s["tokens"]})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", default="models/xlmr_sabty_ner")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--env_file", default=None)
    args = ap.parse_args()

    sents = to_type_dataset(load_conll(TEST))[:args.limit]
    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir, device=args.device, tag_normalizer=tag_to_type)
    if not _load_env(args.env_file) or not os.environ.get("OPENAI_API_KEY"):
        print("[!] no OPENAI_API_KEY — pass --env_file"); return
    client = OpenAIClient(model=args.llm_model, max_tokens=1500)
    positional = LLMNERAgent(client, output_slot="contextual")
    span = LLMNERSpanAgent(client, output_slot="contextual", descriptions=_DESC)

    g_esc, prim_esc, pos_esc, span_esc = [], [], [], []
    escalated = 0
    for s in sents:
        st = _fresh(s, args.threshold)
        tagger.run(st)
        prim = [t.tag for t in st.ner_model_output.sequence_output.tags]
        confs = [t.confidence for t in st.ner_model_output.sequence_output.tags]
        if confs and min(confs) >= args.threshold:
            continue
        escalated += 1
        g_esc.extend(s["tags"]); prim_esc.extend(prim)
        st_p = _fresh(s, args.threshold); positional.run(st_p)
        pos_esc.extend([t.tag for t in st_p.contextual_output.sequence_output.tags])
        st_s = _fresh(s, args.threshold); span.run(st_s)
        span_esc.extend([t.tag for t in st_s.contextual_output.sequence_output.tags])

    print(f"\nEscalated (hard) sentences: {escalated}/{len(sents)}  "
          f"LLM calls: {client.usage_summary()['calls']}  cost≈${client.usage_summary()['est_cost_usd']:.4f}")
    print("\n=== Standalone quality on the ESCALATED sentences (macro-F1 vs gold) ===")
    for name, pred in [("primary (XLM-R)", prim_esc),
                       ("LLM positional (drifts)", pos_esc),
                       ("LLM span (deterministic align)", span_esc)]:
        m, misc = _macro(g_esc, pred)
        print(f"  {name:<32} macroF1={m:.3f}  MISC={misc:.2f}")
    dm = _macro(g_esc, span_esc)[0] - _macro(g_esc, pos_esc)[0]
    print(f"\n  span - positional = {dm:+.3f}  "
          f"({'drift was a real factor' if dm > 0.02 else 'drift NOT the main cause -> genuine disagreement'})")


if __name__ == "__main__":
    main()
