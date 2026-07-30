"""scripts/ner_routing_probe.py

For each synthetic sentence, run the fused NER pipeline and record the router
decision (accept_primary vs escalate) and the min token confidence, at several
thresholds. Shows how many cases actually escalate to the specialist agents.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.transformer_ner_tagger import TransformerNERTagger
from src.state.schema import PipelineState, StateMetadata, TaskConfig

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
DATA = Path(__file__).resolve().parent.parent / "data" / "dev_dummy_ner.jsonl"
THRESHOLDS = [0.5, 0.9, 0.95, 0.99]


def main():
    data = [json.loads(l) for l in DATA.open(encoding="utf-8")]
    print("Loading XLM-R NER model ...")
    tagger = TransformerNERTagger.from_pretrained(device="cpu")

    # Compute the min token confidence per sentence once (router input).
    min_confs = []
    for d in data:
        tokens = d["tokens"]
        seq = tagger.tag(tokens, task_labels=LABELS)
        min_confs.append(min(t.confidence for t in seq.tags))

    print(f"\nMin-token-confidence across {len(data)} sentences:")
    print(f"  lowest={min(min_confs):.3f}  highest={max(min_confs):.3f}  "
          f"mean={sum(min_confs)/len(min_confs):.3f}")

    print(f"\n{'threshold':>10}{'accept':>10}{'escalate':>10}")
    print("-" * 30)
    for th in THRESHOLDS:
        escalate = sum(1 for mc in min_confs if mc < th)
        accept = len(min_confs) - escalate
        print(f"{th:>10.2f}{accept:>10}{escalate:>10}")

    # Show the least-confident sentences (first to escalate).
    order = sorted(range(len(data)), key=lambda i: min_confs[i])[:5]
    print("\nLeast-confident sentences (these escalate first):")
    for i in order:
        print(f"  min_conf={min_confs[i]:.3f}  {data[i]['id']}: {data[i]['text']}")


if __name__ == "__main__":
    main()
