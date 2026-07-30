"""scripts/evaluate_ner_real.py

Evaluate the XLM-R NER primary on the REAL Sabty Arabic-English CS NER corpus
(data/NER/Test_AR-EN_NER.txt). Baseline only — no LLM, no cost.

Because the corpus uses IO tags with 4 types (PERS/LOC/ORG/MISC) while XLM-R
emits IOB2 with 3 types (PER/ORG/LOC, no MISC), both gold and predictions are
reduced to TYPE level ({O, PERS, LOC, ORG, MISC}) for a fair token-level
comparison. XLM-R is expected to score ~0 on MISC (it cannot produce it) — the
gap the LLM agent is meant to fill.

Usage
-----
    python scripts/evaluate_ner_real.py                 # full test set
    python scripts/evaluate_ner_real.py --limit 150     # quick subset
    python scripts/evaluate_ner_real.py --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import _per_class_metrics
from src.evaluation.ner_conll_loader import (
    ENTITY_TYPES, TYPE_LABELS, load_conll, tag_to_type,
)
from src.models.transformer_ner_tagger import TransformerNERTagger

TEST = Path(__file__).resolve().parent.parent / "data" / "NER" / "Test_AR-EN_NER.txt"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Davlan/xlm-roberta-base-ner-hrl")
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="0 = full test set")
    args = ap.parse_args()

    sents = load_conll(TEST)
    if args.limit:
        sents = sents[:args.limit]
    print(f"Loaded {len(sents)} test sentences from {TEST.name}")

    print(f"Loading XLM-R: {args.model_dir or args.model} ...")
    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir or args.model, device=args.device)

    gold_all, pred_all = [], []
    t0 = time.time()
    for i, s in enumerate(sents):
        # XLM-R with its native label set; reduce to type level afterwards.
        pred = tagger.tag(s["tokens"], task_labels=None)
        pred_types = [tag_to_type(tt.tag) for tt in pred.tags]
        gold_types = [tag_to_type(t) for t in s["tags"]]
        n = min(len(gold_types), len(pred_types))
        gold_all.extend(gold_types[:n])
        pred_all.extend(pred_types[:n])
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(sents)} ({time.time()-t0:.0f}s)")

    per = _per_class_metrics(gold_all, pred_all, TYPE_LABELS)
    by = {m.label: m for m in per}
    tok_acc = sum(g == p for g, p in zip(gold_all, pred_all)) / len(gold_all)
    ent_f1s = [by[t].f1 for t in ENTITY_TYPES if t in by]
    macro = sum(ent_f1s) / len(ent_f1s) if ent_f1s else 0.0

    print("\n" + "=" * 58)
    print(f"XLM-R baseline on REAL Sabty test ({len(sents)} sentences)")
    print("=" * 58)
    print(f"  token_accuracy         = {tok_acc:.3f}")
    print(f"  macro_f1 (4 ent types) = {macro:.3f}")
    print(f"\n  {'type':<8}{'P':>7}{'R':>7}{'F1':>7}{'support':>9}")
    for t in TYPE_LABELS:
        if t in by:
            m = by[t]
            print(f"  {t:<8}{m.precision:>7.2f}{m.recall:>7.2f}{m.f1:>7.2f}{m.support:>9}")
    print("\n  NOTE: XLM-R has no MISC label -> expect MISC F1 ~ 0 (the agent's gap).")


if __name__ == "__main__":
    main()
