"""scripts/evaluate_ner_fewshot.py

Test the last untried lever: FEW-SHOT examples from the train set, to teach the
LLM the corpus's annotation conventions (what counts as MISC, that clubs are ORG,
etc.) — targeting the ~0.23 "genuine disagreement" part of the LLM's deficit.

Compares, on the REAL escalated (hard) test sentences:
  primary (XLM-R)     — the trained baseline
  span ZERO-shot      — LLM span extraction, no examples
  span FEW-shot       — same, with K train examples in the prompt

If few-shot >> zero-shot, in-context conventions help; if flat, the gap is
truly training-bound.

    python scripts/evaluate_ner_fewshot.py --model_dir models/xlmr_sabty_ner \\
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

from src.agents.llm_ner_span_agent import LLMNERSpanAgent, spans_from_tags
from src.evaluation.evaluator import _per_class_metrics
from src.evaluation.ner_conll_loader import ENTITY_TYPES, TYPE_LABELS, load_conll, tag_to_type, to_type_dataset
from src.llm.openai_client import OpenAIClient
from src.models.transformer_ner_tagger import TransformerNERTagger
from src.pipeline.ner_agentic import agentic_ner_task_config
from src.state.schema import PipelineState, StateMetadata

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "NER" / "Train_AR-EN_NER.txt"
TEST = ROOT / "data" / "NER" / "Test_AR-EN_NER.txt"
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
            p = ROOT / c
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


def pick_examples(train, k_per_type=2, max_len=22):
    """Pick short, diverse train sentences covering each entity type."""
    picked, seen = [], set()
    for want in ("MISC", "ORG", "PERS", "LOC"):
        cnt = 0
        for i, s in enumerate(train):
            if i in seen or len(s["tokens"]) > max_len:
                continue
            ents = spans_from_tags(s["tokens"], [tag_to_type(t) for t in s["tags"]])
            if ents and any(e["type"] == want for e in ents):
                picked.append((s["text"], ents)); seen.add(i); cnt += 1
                if cnt >= k_per_type:
                    break
    return picked


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

    train = load_conll(TRAIN)
    examples = pick_examples(train)
    print(f"Few-shot examples selected: {len(examples)}")
    for txt, ents in examples:
        print(f"  - {txt[:60]}...  -> {[(e['text'][:14], e['type']) for e in ents]}")

    sents = to_type_dataset(load_conll(TEST))[:args.limit]
    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir, device=args.device, tag_normalizer=tag_to_type)
    if not _load_env(args.env_file) or not os.environ.get("OPENAI_API_KEY"):
        print("[!] no OPENAI_API_KEY — pass --env_file"); return
    client = OpenAIClient(model=args.llm_model, max_tokens=1500)
    zero = LLMNERSpanAgent(client, output_slot="contextual", descriptions=_DESC)
    few = LLMNERSpanAgent(client, output_slot="contextual", descriptions=_DESC, examples=examples)

    g, prim, z, f = [], [], [], []
    escalated = 0
    for s in sents:
        st = _fresh(s, args.threshold); tagger.run(st)
        p = [t.tag for t in st.ner_model_output.sequence_output.tags]
        confs = [t.confidence for t in st.ner_model_output.sequence_output.tags]
        if confs and min(confs) >= args.threshold:
            continue
        escalated += 1
        g.extend(s["tags"]); prim.extend(p)
        stz = _fresh(s, args.threshold); zero.run(stz)
        z.extend([t.tag for t in stz.contextual_output.sequence_output.tags])
        stf = _fresh(s, args.threshold); few.run(stf)
        f.extend([t.tag for t in stf.contextual_output.sequence_output.tags])

    print(f"\nEscalated (hard): {escalated}/{len(sents)}  "
          f"LLM calls: {client.usage_summary()['calls']}  cost≈${client.usage_summary()['est_cost_usd']:.4f}")
    print("\n=== macro-F1 (and MISC) on the ESCALATED sentences ===")
    for name, pred in [("primary (XLM-R)", prim), ("span ZERO-shot", z), ("span FEW-shot", f)]:
        m, misc = _macro(g, pred)
        print(f"  {name:<22} macroF1={m:.3f}  MISC={misc:.2f}")
    d = _macro(g, f)[0] - _macro(g, z)[0]
    print(f"\n  few-shot - zero-shot = {d:+.3f}  "
          f"({'few-shot helps' if d > 0.02 else 'few-shot does NOT close the gap -> training-bound'})")


if __name__ == "__main__":
    main()
