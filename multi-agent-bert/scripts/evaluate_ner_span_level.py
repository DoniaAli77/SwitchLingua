"""scripts/evaluate_ner_span_level.py

Entity/SPAN-level F1 (seqeval) for the fine-tuned XLM-R on the REAL Sabty test
set — the metric Sabty reports (~77.7%), so this makes the comparison fair.
Entity-level = a prediction counts only if the WHOLE span AND type are correct
(stricter than token-level).

Reports per-type + micro + macro entity F1, plus token-level for reference.

    python scripts/evaluate_ner_span_level.py --model_dir models/xlmr_sabty_ner
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

from src.evaluation.ner_conll_loader import load_conll, tag_to_type, to_type_dataset
from src.models.transformer_ner_tagger import TransformerNERTagger

TEST = Path(__file__).resolve().parent.parent / "data" / "NER" / "Test_AR-EN_NER.txt"


def to_bio(type_tags):
    """Convert type-level tags (PERS/LOC/.../O, IO-merged) to BIO for seqeval."""
    out, prev = [], "O"
    for t in type_tags:
        if t == "O":
            out.append("O")
        elif t == prev:
            out.append(f"I-{t}")
        else:
            out.append(f"B-{t}")
        prev = t
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Davlan/xlm-roberta-base-ner-hrl")
    ap.add_argument("--model_dir", default="models/xlmr_sabty_ner")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="0 = full test set")
    args = ap.parse_args()

    from seqeval.metrics import classification_report, f1_score

    sents = to_type_dataset(load_conll(TEST))
    if args.limit:
        sents = sents[:args.limit]
    print(f"Loaded {len(sents)} real test sentences.")
    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir or args.model, device=args.device, tag_normalizer=tag_to_type)

    gold_bio, pred_bio = [], []
    tok_correct = tok_total = 0
    for s in sents:
        pred_types = [tag_to_type(t.tag) for t in tagger.tag(s["tokens"], task_labels=None).tags]
        n = min(len(s["tags"]), len(pred_types))
        g, p = s["tags"][:n], pred_types[:n]
        gold_bio.append(to_bio(g))
        pred_bio.append(to_bio(p))
        tok_correct += sum(a == b for a, b in zip(g, p)); tok_total += n

    print(f"\n{'='*56}\nENTITY / SPAN-LEVEL results (seqeval) — fine-tuned XLM-R\n{'='*56}")
    print(classification_report(gold_bio, pred_bio, digits=3))
    print(f"micro entity-F1 = {f1_score(gold_bio, pred_bio):.3f}")
    print(f"\n(token-level accuracy for reference = {tok_correct/tok_total:.3f})")
    print("\nSabty reference (entity-level): best 0.777 (BiLSTM-CRF), 0.794 (KERMIT++).")


if __name__ == "__main__":
    main()
