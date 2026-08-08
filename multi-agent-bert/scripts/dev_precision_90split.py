"""scripts/dev_precision_90split.py

Per-type PRECISION for the 90%-trained strong primary, evaluated ONLY on its
own held-out 10% dev split (indices saved by finetune_ner_real_90split.py in
split_indices.json). This model never trained on those sentences, and the real
TEST file is never touched -> safe basis for a competence-routing rule.

    python scripts/dev_precision_90split.py --model_dir models/xlmr_sabty_ner_90split
"""

from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--model_dir", default="models/xlmr_sabty_ner_90split")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from seqeval.metrics import classification_report

    split_path = Path(args.model_dir) / "split_indices.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    dev_idx = set(split["dev_idx"])

    all_train = to_type_dataset(load_conll(TRAIN))
    sents = [s for i, s in enumerate(all_train) if i in dev_idx]
    print(f"Held-out dev split: {len(sents)} sentences (seed={split['seed']}, "
          f"never seen by {args.model_dir} during training).")

    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir, device=args.device, tag_normalizer=tag_to_type)

    g_bio, p_bio = [], []
    for i, s in enumerate(sents):
        toks = s["tokens"]
        pred = [t.tag for t in tagger.tag(toks, task_labels=TYPE_LABELS).tags]
        g_bio.append(to_bio(s["tags"]))
        p_bio.append(to_bio(pred))
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(sents)}")

    report = classification_report(g_bio, p_bio, output_dict=True, zero_division=0)
    print(f"\n{'='*60}\nPer-type PRECISION on HELD-OUT DEV — {args.model_dir}\n{'='*60}")
    print(f"{'type':<10}{'precision':>12}{'recall':>10}{'f1':>10}{'support':>10}")
    for etype in ["PERS", "LOC", "ORG", "MISC"]:
        r = report.get(etype, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
        print(f"{etype:<10}{r['precision']:>12.3f}{r['recall']:>10.3f}{r['f1-score']:>10.3f}{r['support']:>10}")
    micro = report["micro avg"]
    print(f"{'micro avg':<10}{micro['precision']:>12.3f}{micro['recall']:>10.3f}{micro['f1-score']:>10.3f}{micro['support']:>10}")
    print("=" * 60)


if __name__ == "__main__":
    main()
