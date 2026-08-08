"""scripts/ner_confidence_distribution.py

LABEL-FREE threshold selection: look at the SHAPE of the primary's own
per-sentence min-token-confidence distribution (the exact statistic
`_ner_route` uses) — no gold tags involved anywhere, so this can be run
directly on the real test set without any leakage concern.

Reports percentiles of min-token-confidence and, for a few candidate
thresholds, what escalation % each would produce. Pick a threshold by where
the distribution naturally breaks, not by checking which one scores best.

    python scripts/ner_confidence_distribution.py --model_dir models/xlmr_sabty_ner --limit 1219
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
TEST = ROOT / "data" / "NER" / "Test_AR-EN_NER.txt"


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", default="models/xlmr_sabty_ner")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=1219)
    args = ap.parse_args()

    sents = to_type_dataset(load_conll(TEST))[: args.limit]
    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir, device=args.device, tag_normalizer=tag_to_type)

    min_confs = []
    for i, s in enumerate(sents):
        out = tagger.tag(s["tokens"], task_labels=TYPE_LABELS)
        confs = [t.confidence for t in out.tags]
        if confs:
            min_confs.append(min(confs))
        if (i + 1) % 300 == 0:
            print(f"  ...{i+1}/{len(sents)}")

    min_confs.sort()
    n = len(min_confs)
    print(f"\nPrimary: {args.model_dir}  |  sentences: {n}  (NO GOLD LABELS USED)")
    print("\nPercentiles of per-sentence MIN-TOKEN-CONFIDENCE:")
    for p in [5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95]:
        v = percentile(min_confs, p)
        print(f"  p{p:>2}: {v:.4f}")

    print("\nEscalation rate at candidate thresholds (escalate if min_conf < threshold):")
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        rate = sum(1 for c in min_confs if c < t) / n
        print(f"  threshold {t:.2f}: escalates {rate:.1%}")


if __name__ == "__main__":
    main()
