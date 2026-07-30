"""scripts/evaluate_ner_retrieval.py

Test the two new agent roles (verification + gazetteer retrieval) on the REAL
escalated test sentences, as a cumulative ladder built on the best agentic
pipeline so far (span + few-shot):

  primary (fine-tuned XLM-R)          — reference
  span + few-shot                     — best agentic so far
  + verify   (LLM yes/no filter)      — targets over-tagging (cause #3)
  + gazetteer (augment, from train)   — targets domain entities (cause #4)
  gazetteer ALONE (deterministic)     — retrieval-only baseline

    python scripts/evaluate_ner_retrieval.py --model_dir models/xlmr_sabty_ner \\
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
from src.agents.ner_retrieval_agents import (
    LLMNERVerifyAgent, NERGazetteerAgent, build_gazetteer_from_conll,
)
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
    "PERS": "a person's name (Arabic or English)", "LOC": "a location/city/country/place",
    "ORG": "an organization/company/team",
    "MISC": "a named entity NOT person/location/organization (events, nationalities, products, competitions, titles)",
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


def _pick_examples(train, k=2, max_len=22):
    picked, seen = [], set()
    for want in ("MISC", "ORG", "PERS", "LOC"):
        cnt = 0
        for i, s in enumerate(train):
            if i in seen or len(s["tokens"]) > max_len:
                continue
            ents = spans_from_tags(s["tokens"], [tag_to_type(t) for t in s["tags"]])
            if ents and any(e["type"] == want for e in ents):
                picked.append((s["text"], ents)); seen.add(i); cnt += 1
                if cnt >= k:
                    break
    return picked


def _fresh(s, thr):
    return PipelineState(metadata=StateMetadata(sample_id=s["id"]), input_text=s["text"],
        task_config=agentic_ner_task_config(TYPE_LABELS, threshold=thr, label_descriptions=_DESC),
        extras={"tokens": s["tokens"]})


def _tags(st):
    return [t.tag for t in st.contextual_output.sequence_output.tags]


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
    gazetteer = build_gazetteer_from_conll(train, tag_to_type)
    print(f"Gazetteer entries (from train): {len(gazetteer)}")
    examples = _pick_examples(train)

    sents = to_type_dataset(load_conll(TEST))[:args.limit]
    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir, device=args.device, tag_normalizer=tag_to_type)
    if not _load_env(args.env_file) or not os.environ.get("OPENAI_API_KEY"):
        print("[!] no OPENAI_API_KEY — pass --env_file"); return
    client = OpenAIClient(model=args.llm_model, max_tokens=1500)
    span = LLMNERSpanAgent(client, output_slot="contextual", descriptions=_DESC, examples=examples)
    verify = LLMNERVerifyAgent(client, source_slot="contextual", output_slot="contextual")
    gaz_aug = NERGazetteerAgent(gazetteer, source_slot="contextual", output_slot="contextual", mode="augment")
    gaz_alone = NERGazetteerAgent(gazetteer, source_slot=None, output_slot="contextual")

    cols = ["primary", "span_fewshot", "+verify", "+gazetteer", "gaz_alone"]
    g = []
    acc = {c: [] for c in cols}
    escalated = 0
    for s in sents:
        st = _fresh(s, args.threshold); tagger.run(st)
        prim = [t.tag for t in st.ner_model_output.sequence_output.tags]
        confs = [t.confidence for t in st.ner_model_output.sequence_output.tags]
        if confs and min(confs) >= args.threshold:
            continue
        escalated += 1
        g.extend(s["tags"]); acc["primary"].extend(prim)

        st1 = _fresh(s, args.threshold); span.run(st1)
        acc["span_fewshot"].extend(_tags(st1))
        verify.run(st1)                        # chain: span -> verify
        acc["+verify"].extend(_tags(st1))
        gaz_aug.run(st1)                        # chain: -> gazetteer augment
        acc["+gazetteer"].extend(_tags(st1))

        stg = _fresh(s, args.threshold); gaz_alone.run(stg)
        acc["gaz_alone"].extend(_tags(stg))

    print(f"\nEscalated (hard): {escalated}/{len(sents)}  "
          f"LLM calls: {client.usage_summary()['calls']}  cost≈${client.usage_summary()['est_cost_usd']:.4f}")
    print("\n" + "=" * 50)
    print(f"{'config':<26}{'macroF1':>10}{'MISC':>8}")
    print("-" * 50)
    for c in cols:
        m, misc = _macro(g, acc[c])
        print(f"{c:<26}{m:>10.3f}{misc:>8.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
