"""Convert the normalized EESA sentiment CSV splits to JSONL.

The Multi-Agent BERT classification path (evaluate_pipeline.py) loads JSONL,
while the EESA processed files are CSV. This converter produces JSONL copies
that preserve the official train/dev/test splits and the exact label set.

Input  (data/Sentiment/processed/):
    eesa_sentiment_train.csv
    eesa_sentiment_dev.csv
    eesa_sentiment_test.csv

Output (same directory):
    eesa_sentiment_{train,dev,test}.jsonl

Schema (one JSON object per line):
    {"text": "...", "label": "positive|negative|neutral"}

Rules:
- official splits preserved (1:1 file mapping, no re-splitting)
- labels copied exactly (validated against the target set)
- only rows with empty text are dropped (and reported)
- original CSVs are never modified

Run:
    python scripts/convert_eesa_csv_to_jsonl.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

PROC_DIR = Path("data/Sentiment/processed")

SPLITS = {
    "eesa_sentiment_train.csv": "eesa_sentiment_train.jsonl",
    "eesa_sentiment_dev.csv": "eesa_sentiment_dev.jsonl",
    "eesa_sentiment_test.csv": "eesa_sentiment_test.jsonl",
}

VALID_LABELS = {"positive", "negative", "neutral"}


def convert(csv_path: Path, jsonl_path: Path) -> dict:
    written = 0
    dropped_empty = 0
    invalid_label = 0
    counts: dict[str, int] = {}

    with open(csv_path, encoding="utf-8", newline="") as fh_in, \
            open(jsonl_path, "w", encoding="utf-8") as fh_out:
        reader = csv.DictReader(fh_in)
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not text:
                dropped_empty += 1
                continue
            if label not in VALID_LABELS:
                invalid_label += 1
                continue
            fh_out.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")
            written += 1
            counts[label] = counts.get(label, 0) + 1

    return {
        "written": written,
        "dropped_empty": dropped_empty,
        "invalid_label": invalid_label,
        "labels": counts,
    }


def main() -> None:
    for csv_name, jsonl_name in SPLITS.items():
        csv_path = PROC_DIR / csv_name
        jsonl_path = PROC_DIR / jsonl_name
        if not csv_path.exists():
            print(f"SKIP {csv_name}: not found")
            continue
        stats = convert(csv_path, jsonl_path)
        print(
            f"{jsonl_name}: {stats['written']} rows  "
            f"labels={stats['labels']}  "
            f"dropped_empty={stats['dropped_empty']}  "
            f"invalid_label={stats['invalid_label']}"
        )


if __name__ == "__main__":
    main()
