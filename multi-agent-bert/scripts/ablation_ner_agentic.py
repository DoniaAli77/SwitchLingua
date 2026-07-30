"""scripts/ablation_ner_agentic.py

Configurable ABLATION over the agentic-augmentation components, to find the best
combination. All augmentations build ON the fine-tuned primary (never replace it):

  gaz    : gazetteer retrieval (built from TRAIN) fills the primary's O gaps
  llm    : LLM span extraction (few-shot from TRAIN) proposes entities in O gaps
  verify : LLM confirms proposed additions; only confirmed ones are applied

Sweeps all valid combinations and reports entity-level F1/P/R (seqeval) on the
test set, then names the best combo.

NOTHING is hardcoded to this corpus:
  * label set is derived from the TRAIN file
  * gazetteer + few-shot examples are built from TRAIN
  * type descriptions are generic ("a named entity of type X")
  * threshold / model / sizes are CLI args
Per-sentence, each component is computed ONCE and reused across combos (cheap).

    python scripts/ablation_ner_agentic.py --model_dir models/xlmr_sabty_ner \\
        --limit 300 --env_file ../Modified_Version/.env
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

from src.agents.llm_ner_agent import coerce_to_valid
from src.agents.llm_ner_span_agent import (
    LLMNERSpanAgent, _norm, align_entities_to_tokens, build_span_prompt, spans_from_tags,
)
from src.agents.ner_retrieval_agents import LLMNERVerifyAgent, build_gazetteer_from_conll
from src.evaluation.ner_conll_loader import load_conll, tag_to_type, to_type_dataset
from src.llm.openai_client import OpenAIClient
from src.models.transformer_ner_tagger import TransformerNERTagger

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "NER" / "Train_AR-EN_NER.txt"
TEST = ROOT / "data" / "NER" / "Test_AR-EN_NER.txt"


def to_bio(type_tags):
    out, prev = [], "O"
    for t in type_tags:
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


def derive_labels(train):
    """Label set from the DATA (not hardcoded): O + observed entity types."""
    types = set()
    for s in train:
        for t in s["tags"]:
            ty = tag_to_type(t)
            if ty != "O":
                types.add(ty)
    return ["O"] + sorted(types)


def pick_examples(train, labels, k=2, max_len=22):
    ent_types = [l for l in labels if l != "O"]
    picked, seen = [], set()
    for want in ent_types:
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


def gap_spans(tokens, prim, cand_tags):
    """Spans where cand_tags marks an entity AND primary is O (additions)."""
    spans, i, n = [], 0, len(tokens)
    while i < n:
        if prim[i] == "O" and cand_tags[i] != "O":
            ty, j = cand_tags[i], i
            while j < n and prim[j] == "O" and cand_tags[j] == ty:
                j += 1
            spans.append({"start": i, "end": j, "type": ty, "text": " ".join(tokens[i:j])})
            i = j
        else:
            i += 1
    return spans


def gaz_tags(tokens, gazetteer, valid):
    norm = [_norm(t) for t in tokens]
    tags = ["O"] * len(tokens)
    maxlen = max((len(k.split()) for k in gazetteer), default=1)
    i = 0
    while i < len(tokens):
        hit = False
        for L in range(min(maxlen, len(tokens) - i), 0, -1):
            key = " ".join(norm[i:i + L])
            if key in gazetteer:
                ty = coerce_to_valid(gazetteer[key], valid)
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


def _key(text):
    return " ".join(_norm(x) for x in text.split())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", default="models/xlmr_sabty_ner")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--env_file", default=None)
    args = ap.parse_args()

    from seqeval.metrics import f1_score, precision_score, recall_score

    train = load_conll(TRAIN)
    labels = derive_labels(train)
    valid = set(labels)
    print(f"Labels derived from train: {labels}")
    gazetteer = build_gazetteer_from_conll(train, tag_to_type)
    examples = pick_examples(train, labels)
    descriptions = {t: f"a named entity of type {t}" for t in labels if t != "O"}

    sents = to_type_dataset(load_conll(TEST))
    if args.limit:
        sents = sents[:args.limit]
    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir, device=args.device, tag_normalizer=tag_to_type)
    if not _load_env(args.env_file) or not os.environ.get("OPENAI_API_KEY"):
        print("[!] no OPENAI_API_KEY — pass --env_file"); return
    client = OpenAIClient(model=args.llm_model, max_tokens=1500)

    combos = ["primary", "gaz", "llm", "gaz+llm",
              "gaz+verify", "llm+verify", "gaz+llm+verify"]
    gold_bio = []
    preds = {c: [] for c in combos}

    for s in sents:
        toks = s["tokens"]
        prim = [tag_to_type(t.tag) for t in tagger.tag(toks, task_labels=None).tags]
        n = min(len(s["tags"]), len(prim))
        gold, prim, toks = s["tags"][:n], prim[:n], toks[:n]
        gold_bio.append(to_bio(gold))

        # component 1: gazetteer additions (in O gaps)
        gaz_adds = gap_spans(toks, prim, gaz_tags(toks, gazetteer, valid))

        # component 2: LLM span additions (few-shot), in O gaps
        raw = client.generate(build_span_prompt("ner", labels, toks, s["text"],
                                                descriptions, examples))
        try:
            ents, _ = LLMNERSpanAgent._parse(raw)
        except Exception:
            ents = []
        llm_full = align_entities_to_tokens(toks, ents, valid)
        llm_adds = gap_spans(toks, prim, llm_full)

        # component 3: verify the UNION of additions once; keep confirmed
        union = gaz_adds + [a for a in llm_adds
                            if not any(a["start"] == g["start"] and a["end"] == g["end"] for g in gaz_adds)]
        confirmed_keys = set()
        if union:
            vraw = client.generate(LLMNERVerifyAgent._prompt(s["text"], union))
            try:
                kept = LLMNERVerifyAgent._parse_kept(vraw, union)
                confirmed_keys = {_key(e["text"]) for e in kept}
            except Exception:
                confirmed_keys = {_key(a["text"]) for a in union}  # keep all on parse fail
        ver = lambda spans: [a for a in spans if _key(a["text"]) in confirmed_keys]

        # assemble the combos from cached pieces
        asm = {
            "primary": prim,
            "gaz": apply_spans(prim, gaz_adds),
            "llm": apply_spans(prim, llm_adds),
            "gaz+llm": apply_spans(prim, gaz_adds + llm_adds),
            "gaz+verify": apply_spans(prim, ver(gaz_adds)),
            "llm+verify": apply_spans(prim, ver(llm_adds)),
            "gaz+llm+verify": apply_spans(prim, ver(gaz_adds + llm_adds)),
        }
        for c in combos:
            preds[c].append(to_bio(asm[c]))

    print(f"\nSentences: {len(sents)}  LLM calls: {client.usage_summary()['calls']}"
          f"  cost≈${client.usage_summary()['est_cost_usd']:.4f}")
    print("\n" + "=" * 58)
    print(f"{'combination':<20}{'F1':>8}{'precision':>11}{'recall':>9}")
    print("-" * 58)
    scored = []
    for c in combos:
        f1 = f1_score(gold_bio, preds[c])
        p = precision_score(gold_bio, preds[c]); r = recall_score(gold_bio, preds[c])
        scored.append((c, f1, p, r))
        print(f"{c:<20}{f1:>8.3f}{p:>11.3f}{r:>9.3f}")
    print("=" * 58)
    best = max(scored, key=lambda x: x[1])
    print(f"\nBEST combination: '{best[0]}'  (F1={best[1]:.3f})")


if __name__ == "__main__":
    main()
