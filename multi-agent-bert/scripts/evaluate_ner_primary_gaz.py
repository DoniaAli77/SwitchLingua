"""scripts/evaluate_ner_primary_gaz.py

Can agents BEAT the fine-tuned baseline (not just approach it)? The trick: don't
REPLACE the strong primary — AUGMENT it. Keep every primary prediction and let
the gazetteer (retrieval from train) fill only the primary's O gaps + add missed
entities. Deterministic, no LLM, evaluated on the FULL test set at entity level
(seqeval) — directly comparable to the 0.816 baseline.

    python scripts/evaluate_ner_primary_gaz.py --model_dir models/xlmr_sabty_ner
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.ner_retrieval_agents import build_gazetteer_from_conll
from src.agents.llm_ner_span_agent import _norm
from src.agents.llm_ner_agent import coerce_to_valid
from src.evaluation.evaluator import _per_class_metrics
from src.evaluation.ner_conll_loader import ENTITY_TYPES, TYPE_LABELS, load_conll, tag_to_type, to_type_dataset
from src.models.transformer_ner_tagger import TransformerNERTagger


def to_bio(type_tags):
    """Type-level tags (IO-merged) -> BIO for seqeval entity-level scoring."""
    out, prev = [], "O"
    for t in type_tags:
        out.append("O" if t == "O" else (f"I-{t}" if t == prev else f"B-{t}"))
        prev = t
    return out

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "NER" / "Train_AR-EN_NER.txt"
TEST = ROOT / "data" / "NER" / "Test_AR-EN_NER.txt"


def gaz_augment(tokens, prim_tags, gazetteer, valid):
    """Fill only the primary's O positions with gazetteer matches."""
    norm = [_norm(t) for t in tokens]
    out = list(prim_tags)
    maxlen = max((len(k.split()) for k in gazetteer), default=1)
    i = 0
    while i < len(tokens):
        hit = False
        for L in range(min(maxlen, len(tokens) - i), 0, -1):
            # only augment if ALL positions in the span are currently O
            if any(out[i + j] != "O" for j in range(L)):
                continue
            key = " ".join(norm[i:i + L])
            if key in gazetteer:
                ty = coerce_to_valid(gazetteer[key], valid)
                if ty != "O":
                    for j in range(L):
                        out[i + j] = ty
                    i += L; hit = True; break
        if not hit:
            i += 1
    return out


def report(name, gold_bio, pred_bio, gold_tok, pred_tok):
    from seqeval.metrics import f1_score, precision_score, recall_score
    per = {m.label: m for m in _per_class_metrics(gold_tok, pred_tok, TYPE_LABELS)}
    tokmac = sum(per[t].f1 for t in ENTITY_TYPES if t in per) / len(ENTITY_TYPES)
    print(f"\n=== {name} ===")
    print(f"  entity-F1 (seqeval micro) = {f1_score(gold_bio, pred_bio):.3f}"
          f"  (P={precision_score(gold_bio, pred_bio):.3f} R={recall_score(gold_bio, pred_bio):.3f})")
    print(f"  token-level macro-F1      = {tokmac:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", default="models/xlmr_sabty_ner")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    gazetteer = build_gazetteer_from_conll(load_conll(TRAIN), tag_to_type)
    print(f"Gazetteer entries: {len(gazetteer)}")
    sents = to_type_dataset(load_conll(TEST))
    if args.limit:
        sents = sents[:args.limit]
    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir, device=args.device, tag_normalizer=tag_to_type)

    g_bio, p_bio, pg_bio = [], [], []
    g_tok, p_tok, pg_tok = [], [], []
    for s in sents:
        prim = [tag_to_type(t.tag) for t in tagger.tag(s["tokens"], task_labels=None).tags]
        n = min(len(s["tags"]), len(prim))
        gold, prim = s["tags"][:n], prim[:n]
        toks = s["tokens"][:n]
        aug = gaz_augment(toks, prim, gazetteer, set(TYPE_LABELS))
        g_bio.append(to_bio(gold)); p_bio.append(to_bio(prim)); pg_bio.append(to_bio(aug))
        g_tok += gold; p_tok += prim; pg_tok += aug

    report("PRIMARY alone (baseline)", g_bio, p_bio, g_tok, p_tok)
    report("PRIMARY + gazetteer-augment", g_bio, pg_bio, g_tok, pg_tok)
    print("\nSabty best (entity-level): 0.777 / 0.794")


if __name__ == "__main__":
    main()
