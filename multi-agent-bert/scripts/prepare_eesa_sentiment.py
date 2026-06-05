"""Normalize the raw EESA-Corpus sentiment splits into the Multi-Agent BERT schema.

Raw EESA files (data/Sentiment/raw/EESA-Corpus-main/) are headerless 2-column
CSVs: column 0 = comment text, column 1 = sentiment in {positive, negative,
neutral}. EESA ships official train/dev/test splits, so we PRESERVE them (no
re-splitting) and only normalize to:

    text,label          # header row
    "...",positive

Labels already match the target set, so mapping is identity (with lowercase +
whitespace strip + validation). The original raw files are left untouched.

Run:
    python scripts/prepare_eesa_sentiment.py
"""

from __future__ import annotations

import csv
from pathlib import Path

RAW_DIR = Path("data/Sentiment/raw/EESA-Corpus-main")
OUT_DIR = Path("data/Sentiment/processed")

# raw filename -> normalized output filename
SPLITS = {
    "EESA-Train.csv": "eesa_sentiment_train.csv",
    "EESA-Dev.csv": "eesa_sentiment_dev.csv",
    "EESA-Test.csv": "eesa_sentiment_test.csv",
}

VALID_LABELS = {"positive", "negative", "neutral"}


def normalize_split(raw_path: Path, out_path: Path) -> dict:
    rows: list[tuple[str, str]] = []
    skipped = 0
    with open(raw_path, encoding="utf-8", newline="") as fh:
        for rec in csv.reader(fh):
            if len(rec) < 2:
                skipped += 1
                continue
            text = rec[0].strip()
            label = rec[1].strip().lower()
            if not text or label not in VALID_LABELS:
                skipped += 1
                continue
            rows.append((text, label))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "label"])
        w.writerows(rows)

    counts: dict[str, int] = {}
    for _, label in rows:
        counts[label] = counts.get(label, 0) + 1
    return {"written": len(rows), "skipped": skipped, "labels": counts}


def main() -> None:
    for raw_name, out_name in SPLITS.items():
        raw_path = RAW_DIR / raw_name
        out_path = OUT_DIR / out_name
        stats = normalize_split(raw_path, out_path)
        print(f"{out_name}: {stats['written']} rows  labels={stats['labels']}  skipped={stats['skipped']}")


if __name__ == "__main__":
    main()
