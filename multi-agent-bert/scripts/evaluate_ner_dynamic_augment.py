"""scripts/evaluate_ner_dynamic_augment.py

The best shot at BEATING the baseline: augment the fine-tuned primary with the
STRONGEST agent components — DYNAMIC few-shot LLM proposals + gazetteer — each
filling only the primary's O gaps, then VERIFY every addition.

Configs (full test-subset, entity-level seqeval):
  PRIMARY                          — baseline
  + dynLLM + verify                — dynamic-few-shot LLM additions, verified
  + gaz + verify                   — gazetteer additions, verified
  + dynLLM + gaz + verify          — both, verified  (the full stack)

    python scripts/evaluate_ner_dynamic_augment.py --model_dir models/xlmr_sabty_ner \\
        --limit 300 --env_file ../Modified_Version/.env
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

from src.agents.llm_ner_agent import coerce_to_valid
from src.agents.llm_ner_span_agent import (
    LLMNERSpanAgent, _norm, align_entities_to_tokens, build_span_prompt, spans_from_tags,
)
from src.agents.ner_retrieval_agents import LLMNERVerifyAgent, build_gazetteer_from_conll
from src.evaluation.ner_conll_loader import TYPE_LABELS, load_conll, tag_to_type, to_type_dataset
from src.llm.openai_client import OpenAIClient
from src.models.transformer_ner_tagger import TransformerNERTagger

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "NER" / "Train_AR-EN_NER.txt"
TEST = ROOT / "data" / "NER" / "Test_AR-EN_NER.txt"


def to_bio(tags):
    out, prev = [], "O"
    for t in tags:
        out.append("O" if t == "O" else (f"I-{t}" if t == prev else f"B-{t}"))
        prev = t
    return out


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


class Retriever:
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
        scored = [(sum(self.idf.get(t, 0.0) for t in (q & d["toks"])), d)
                  for d in self.data if q & d["toks"]]
        scored.sort(key=lambda x: -x[0])
        return [(d["text"], d["ents"]) for _, d in scored[:k]]


def gap_spans(tokens, prim, cand):
    spans, i, n = [], 0, len(tokens)
    while i < n:
        if prim[i] == "O" and cand[i] != "O":
            ty, j = cand[i], i
            while j < n and prim[j] == "O" and cand[j] == ty:
                j += 1
            spans.append({"start": i, "end": j, "type": ty, "text": " ".join(tokens[i:j])})
            i = j
        else:
            i += 1
    return spans


def gaz_full(tokens, gaz, valid):
    norm = [_norm(t) for t in tokens]
    tags = ["O"] * len(tokens)
    maxlen = max((len(k.split()) for k in gaz), default=1)
    i = 0
    while i < len(tokens):
        hit = False
        for L in range(min(maxlen, len(tokens) - i), 0, -1):
            key = " ".join(norm[i:i + L])
            if key in gaz:
                ty = coerce_to_valid(gaz[key], valid)
                if ty != "O":
                    for j in range(i, i + L):
                        tags[j] = ty
                    i += L; hit = True; break
        if not hit:
            i += 1
    return tags


def apply_spans(prim, spans):
    out = list(prim)
    for s in spans:
        for j in range(s["start"], s["end"]):
            out[j] = s["type"]
    return out


def _key(t):
    return " ".join(_norm(x) for x in t.split())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", default="models/xlmr_sabty_ner")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--env_file", default=None)
    args = ap.parse_args()

    from seqeval.metrics import f1_score, precision_score, recall_score

    train = load_conll(TRAIN)
    gaz = build_gazetteer_from_conll(train, tag_to_type)
    retr = Retriever(train)
    valid = set(TYPE_LABELS)
    desc = {t: f"a named entity of type {t}" for t in TYPE_LABELS if t != "O"}

    sents = to_type_dataset(load_conll(TEST))[:args.limit]
    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir, device=args.device, tag_normalizer=tag_to_type)
    if not _load_env(args.env_file) or not os.environ.get("OPENAI_API_KEY"):
        print("[!] no OPENAI_API_KEY — pass --env_file"); return
    client = OpenAIClient(model=args.llm_model, max_tokens=1500)

    cfgs = ["primary", "dynLLM_alone", "+dynLLM+verify", "+gaz+verify", "+dynLLM+gaz+verify"]
    g = []
    pred = {c: [] for c in cfgs}
    for s in sents:
        toks = s["tokens"]
        prim = [tag_to_type(t.tag) for t in tagger.tag(toks, task_labels=None).tags]
        n = min(len(s["tags"]), len(prim))
        gold, prim, toks = s["tags"][:n], prim[:n], toks[:n]
        g.append(to_bio(gold))

        # dynamic-few-shot LLM proposals -> additions in O gaps
        raw = client.generate(build_span_prompt("ner", TYPE_LABELS, toks, s["text"],
                                                desc, retr.top(toks, args.k)))
        try:
            ents, _ = LLMNERSpanAgent._parse(raw)
        except Exception:
            ents = []
        llm_full = align_entities_to_tokens(toks, ents, valid)
        dyn_adds = gap_spans(toks, prim, llm_full)
        gaz_adds = gap_spans(toks, prim, gaz_full(toks, gaz, valid))

        union = dyn_adds + [a for a in gaz_adds
                            if not any(a["start"] == d["start"] and a["end"] == d["end"] for d in dyn_adds)]
        conf = set()
        if union:
            vraw = client.generate(LLMNERVerifyAgent._prompt(s["text"], union))
            try:
                conf = {_key(e["text"]) for e in LLMNERVerifyAgent._parse_kept(vraw, union)}
            except Exception:
                conf = {_key(a["text"]) for a in union}
        v = lambda spans: [a for a in spans if _key(a["text"]) in conf]

        pred["primary"].append(to_bio(prim))
        pred["dynLLM_alone"].append(to_bio(llm_full))
        pred["+dynLLM+verify"].append(to_bio(apply_spans(prim, v(dyn_adds))))
        pred["+gaz+verify"].append(to_bio(apply_spans(prim, v(gaz_adds))))
        pred["+dynLLM+gaz+verify"].append(to_bio(apply_spans(prim, v(dyn_adds + gaz_adds))))

    print(f"\nSentences: {len(sents)}  LLM calls: {client.usage_summary()['calls']}"
          f"  cost≈${client.usage_summary()['est_cost_usd']:.4f}")
    print("\n=== entity-level (seqeval) ===")
    base = f1_score(g, pred["primary"])
    for c in cfgs:
        f1 = f1_score(g, pred[c])
        mark = "" if c == "primary" else ("  ← BEATS baseline!" if f1 > base + 0.0005 else
                                          ("  (ties)" if abs(f1 - base) <= 0.0005 else ""))
        print(f"  {c:<22} F1={f1:.3f}  P={precision_score(g, pred[c]):.3f}  R={recall_score(g, pred[c]):.3f}{mark}")


if __name__ == "__main__":
    main()
