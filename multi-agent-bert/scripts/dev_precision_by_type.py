"""scripts/dev_precision_by_type.py

Per-entity-type PRECISION of a primary model on a held-out DEV set (never test).
Used to derive a competence-based routing rule (which types to escalate to
agents) without touching the real test set.

For the gen-240-trained primary, real Sabty TRAIN is a valid dev set: that
model has never seen a single real-train sentence during training. Entity-level
(seqeval) precision/recall/F1 per type, same metric family used throughout.

    python scripts/dev_precision_by_type.py --model_dir models/xlmr_gen240_ner --limit 1500
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

from src.evaluation.ner_conll_loader import TYPE_LABELS, load_conll, tag_to_type, to_type_dataset
from src.models.transformer_ner_tagger import TransformerNERTagger

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "NER" / "Train_AR-EN_NER.txt"


def to_bio(type_tags):
    out, prev = [], "O"
    for t in type_tags:
        out.append("O" if t == "O" else (f"I-{t}" if t == prev else f"B-{t}"))
        prev = t
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", default="models/xlmr_gen240_ner")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None, help="cap #dev sentences (default: all)")
    args = ap.parse_args()

    from seqeval.metrics import classification_report

    sents = to_type_dataset(load_conll(TRAIN))
    if args.limit:
        sents = sents[: args.limit]
    print(f"Dev set: real Sabty TRAIN, {len(sents)} sentences (never seen by {args.model_dir}).")

    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir, device=args.device, tag_normalizer=tag_to_type)

    g_bio, p_bio = [], []
    for i, s in enumerate(sents):
        toks = s["tokens"]
        pred = [t.tag for t in tagger.tag(toks, task_labels=TYPE_LABELS).tags]
        g_bio.append(to_bio(s["tags"]))
        p_bio.append(to_bio(pred))
        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(sents)}")

    report = classification_report(g_bio, p_bio, output_dict=True, zero_division=0)
    print(f"\n{'='*60}\nPer-type PRECISION on DEV (real train) — {args.model_dir}\n{'='*60}")
    print(f"{'type':<10}{'precision':>12}{'recall':>10}{'f1':>10}{'support':>10}")
    for etype in ["PERS", "LOC", "ORG", "MISC"]:
        r = report.get(etype, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
        print(f"{etype:<10}{r['precision']:>12.3f}{r['recall']:>10.3f}{r['f1-score']:>10.3f}{r['support']:>10}")
    micro = report["micro avg"]
    print(f"{'micro avg':<10}{micro['precision']:>12.3f}{micro['recall']:>10.3f}{micro['f1-score']:>10.3f}{micro['support']:>10}")
    print("=" * 60)


if __name__ == "__main__":
    main()
