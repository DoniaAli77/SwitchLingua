"""scripts/evaluate_ner_dynamic_fewshot.py

Test the last untried technique: DYNAMIC (retrieval-based) few-shot. Instead of a
FIXED set of examples for every sentence, retrieve the train sentences most
similar to each test sentence (IDF-weighted token overlap — no extra models) and
use THEIR gold entities as the examples. Idea: a football sentence gets football
examples that teach the club convention.

Compares the LLM span agent's standalone quality on the escalated (hard)
sentences: primary vs FIXED few-shot vs DYNAMIC few-shot.

    python scripts/evaluate_ner_dynamic_fewshot.py --model_dir models/xlmr_sabty_ner \\
        --limit 100 --env_file ../Modified_Version/.env
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.llm_ner_span_agent import (
    LLMNERSpanAgent, _norm, align_entities_to_tokens, build_span_prompt, spans_from_tags,
)
from src.evaluation.evaluator import _per_class_metrics
from src.evaluation.ner_conll_loader import ENTITY_TYPES, TYPE_LABELS, load_conll, tag_to_type, to_type_dataset
from src.llm.openai_client import OpenAIClient
from src.models.transformer_ner_tagger import TransformerNERTagger

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "NER" / "Train_AR-EN_NER.txt"
TEST = ROOT / "data" / "NER" / "Test_AR-EN_NER.txt"


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


class Retriever:
    """IDF-weighted token-overlap retriever over train sentences with entities."""
    def __init__(self, train, max_len=30):
        self.data, df = [], Counter()
        for s in train:
            if len(s["tokens"]) > max_len:
                continue
            ents = spans_from_tags(s["tokens"], [tag_to_type(t) for t in s["tags"]])
            if not ents:
                continue
            toks = {t for t in (_norm(x) for x in s["tokens"]) if t}
            self.data.append({"text": s["text"], "ents": ents, "toks": toks})
            for t in toks:
                df[t] += 1
        n = max(1, len(self.data))
        self.idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}

    def top(self, tokens, k=6):
        q = {t for t in (_norm(x) for x in tokens) if t}
        scored = []
        for d in self.data:
            sh = q & d["toks"]
            if sh:
                scored.append((sum(self.idf.get(t, 0.0) for t in sh), d))
        scored.sort(key=lambda x: -x[0])
        return [(d["text"], d["ents"]) for _, d in scored[:k]]


def _pick_fixed(train, labels, k=2, max_len=22):
    picked, seen = [], set()
    for want in [l for l in labels if l != "O"]:
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


def _span_tags(client, labels, toks, text, examples, valid):
    raw = client.generate(build_span_prompt("ner", labels, toks, text,
                          {t: f"a named entity of type {t}" for t in labels if t != "O"},
                          examples))
    try:
        ents, _ = LLMNERSpanAgent._parse(raw)
    except Exception:
        ents = []
    return align_entities_to_tokens(toks, ents, valid)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", default="models/xlmr_sabty_ner")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--env_file", default=None)
    args = ap.parse_args()

    train = load_conll(TRAIN)
    labels = TYPE_LABELS
    valid = set(labels)
    fixed = _pick_fixed(train, labels)
    retr = Retriever(train)
    print(f"Retriever pool: {len(retr.data)} train sentences with entities.")

    sents = to_type_dataset(load_conll(TEST))[:args.limit]
    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir, device=args.device, tag_normalizer=tag_to_type)
    if not _load_env(args.env_file) or not os.environ.get("OPENAI_API_KEY"):
        print("[!] no OPENAI_API_KEY — pass --env_file"); return
    client = OpenAIClient(model=args.llm_model, max_tokens=1500)

    g, prim, fx, dy = [], [], [], []
    escalated = 0
    for s in sents:
        st = tagger.tag(s["tokens"], task_labels=None)
        p = [tag_to_type(t.tag) for t in st.tags]
        confs = [t.confidence for t in st.tags]
        if confs and min(confs) >= args.threshold:
            continue
        escalated += 1
        toks = s["tokens"]
        g.extend(s["tags"]); prim.extend(p)
        fx.extend(_span_tags(client, labels, toks, s["text"], fixed, valid))
        dy.extend(_span_tags(client, labels, toks, s["text"], retr.top(toks, args.k), valid))

    print(f"\nEscalated (hard): {escalated}/{len(sents)}  "
          f"LLM calls: {client.usage_summary()['calls']}  cost≈${client.usage_summary()['est_cost_usd']:.4f}")
    print("\n=== macro-F1 (and MISC) on ESCALATED sentences ===")
    for name, pred in [("primary (XLM-R)", prim), ("span FIXED few-shot", fx),
                       ("span DYNAMIC few-shot", dy)]:
        m, misc = _macro(g, pred)
        print(f"  {name:<24} macroF1={m:.3f}  MISC={misc:.2f}")
    d = _macro(g, dy)[0] - _macro(g, fx)[0]
    print(f"\n  dynamic - fixed = {d:+.3f}  "
          f"({'dynamic helps' if d > 0.02 else 'dynamic ~ fixed (no real gain)'})")


if __name__ == "__main__":
    main()
