"""scripts/generate_dummy_ner_data.py

Generates ``data/dev_dummy_ner.jsonl`` — 30 Arabic-English code-switched
examples for the ``ner`` task (BIO scheme).

Format (one JSON object per line)
----------------------------------
{
    "id":     "ner_01",
    "text":   "Ahmed works at Google in Cairo.",
    "tokens": ["Ahmed", "works", "at", "Google", "in", "Cairo", "."],
    "tags":   ["B-PER", "O", "O", "B-ORG", "O", "B-LOC", "O"]
}

All samples are manually crafted so len(tokens) == len(tags) is trivially
verifiable by inspection.  Labels used: O, B-PER, I-PER, B-ORG, I-ORG,
B-LOC, I-LOC.

Usage
-----
    python scripts/generate_dummy_ner_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "dev_dummy_ner.jsonl"

# ---------------------------------------------------------------------------
# Manually crafted examples
# Each dict has: id, text, tokens, tags
# ---------------------------------------------------------------------------

_EXAMPLES: list[dict] = [
    # ── Person names ─────────────────────────────────────────────────────
    {
        "id": "ner_01",
        "text": "Ahmed works at Google.",
        "tokens": ["Ahmed", "works", "at", "Google", "."],
        "tags":   ["B-PER", "O", "O", "B-ORG", "O"],
    },
    {
        "id": "ner_02",
        "text": "أحمد يعمل في شركة Microsoft.",
        "tokens": ["أحمد", "يعمل", "في", "شركة", "Microsoft", "."],
        "tags":   ["B-PER", "O", "O", "O", "B-ORG", "O"],
    },
    {
        "id": "ner_03",
        "text": "Sara and Mohamed visited Paris last summer.",
        "tokens": ["Sara", "and", "Mohamed", "visited", "Paris", "last", "summer", "."],
        "tags":   ["B-PER", "O", "B-PER", "O", "B-LOC", "O", "O", "O"],
    },
    {
        "id": "ner_04",
        "text": "سارة ومحمد زارا القاهرة معاً.",
        "tokens": ["سارة", "ومحمد", "زارا", "القاهرة", "معاً", "."],
        "tags":   ["B-PER", "B-PER", "O", "B-LOC", "O", "O"],
    },
    {
        "id": "ner_05",
        "text": "The CEO John Smith announced the merger.",
        "tokens": ["The", "CEO", "John", "Smith", "announced", "the", "merger", "."],
        "tags":   ["O", "O", "B-PER", "I-PER", "O", "O", "O", "O"],
    },
    {
        "id": "ner_06",
        "text": "المدير التنفيذي خالد العمري أعلن عن الاندماج.",
        "tokens": ["المدير", "التنفيذي", "خالد", "العمري", "أعلن", "عن", "الاندماج", "."],
        "tags":   ["O", "O", "B-PER", "I-PER", "O", "O", "O", "O"],
    },
    {
        "id": "ner_07",
        "text": "Dr. Nour Hassan teaches at Cairo University.",
        "tokens": ["Dr.", "Nour", "Hassan", "teaches", "at", "Cairo", "University", "."],
        "tags":   ["O", "B-PER", "I-PER", "O", "O", "B-ORG", "I-ORG", "O"],
    },
    {
        "id": "ner_08",
        "text": "الدكتورة نور حسن تدرّس في جامعة القاهرة.",
        "tokens": ["الدكتورة", "نور", "حسن", "تدرّس", "في", "جامعة", "القاهرة", "."],
        "tags":   ["O", "B-PER", "I-PER", "O", "O", "B-ORG", "I-ORG", "O"],
    },
    {
        "id": "ner_09",
        "text": "Omar السيد joined Amazon in New York.",
        "tokens": ["Omar", "السيد", "joined", "Amazon", "in", "New", "York", "."],
        "tags":   ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC", "I-LOC", "O"],
    },
    {
        "id": "ner_10",
        "text": "Fatima القاضي received an award from UNESCO.",
        "tokens": ["Fatima", "القاضي", "received", "an", "award", "from", "UNESCO", "."],
        "tags":   ["B-PER", "I-PER", "O", "O", "O", "O", "B-ORG", "O"],
    },
    # ── Organization names ───────────────────────────────────────────────
    {
        "id": "ner_11",
        "text": "Google opened a new office in Dubai.",
        "tokens": ["Google", "opened", "a", "new", "office", "in", "Dubai", "."],
        "tags":   ["B-ORG", "O", "O", "O", "O", "O", "B-LOC", "O"],
    },
    {
        "id": "ner_12",
        "text": "افتتحت شركة أمازون مكتباً جديداً في دبي.",
        "tokens": ["افتتحت", "شركة", "أمازون", "مكتباً", "جديداً", "في", "دبي", "."],
        "tags":   ["O", "O", "B-ORG", "O", "O", "O", "B-LOC", "O"],
    },
    {
        "id": "ner_13",
        "text": "The United Nations held a summit in Geneva.",
        "tokens": ["The", "United", "Nations", "held", "a", "summit", "in", "Geneva", "."],
        "tags":   ["O", "B-ORG", "I-ORG", "O", "O", "O", "O", "B-LOC", "O"],
    },
    {
        "id": "ner_14",
        "text": "الأمم المتحدة عقدت قمة في جنيف.",
        "tokens": ["الأمم", "المتحدة", "عقدت", "قمة", "في", "جنيف", "."],
        "tags":   ["B-ORG", "I-ORG", "O", "O", "O", "B-LOC", "O"],
    },
    {
        "id": "ner_15",
        "text": "Apple و Microsoft تتنافسان في السوق.",
        "tokens": ["Apple", "و", "Microsoft", "تتنافسان", "في", "السوق", "."],
        "tags":   ["B-ORG", "O", "B-ORG", "O", "O", "O", "O"],
    },
    {
        "id": "ner_16",
        "text": "OpenAI released a new model in San Francisco.",
        "tokens": ["OpenAI", "released", "a", "new", "model", "in", "San", "Francisco", "."],
        "tags":   ["B-ORG", "O", "O", "O", "O", "O", "B-LOC", "I-LOC", "O"],
    },
    {
        "id": "ner_17",
        "text": "أصدرت OpenAI نموذجاً جديداً من مقرها في San Francisco.",
        "tokens": ["أصدرت", "OpenAI", "نموذجاً", "جديداً", "من", "مقرها", "في", "San", "Francisco", "."],
        "tags":   ["O", "B-ORG", "O", "O", "O", "O", "O", "B-LOC", "I-LOC", "O"],
    },
    {
        "id": "ner_18",
        "text": "Harvard University is located in Cambridge Massachusetts.",
        "tokens": ["Harvard", "University", "is", "located", "in", "Cambridge", "Massachusetts", "."],
        "tags":   ["B-ORG", "I-ORG", "O", "O", "O", "B-LOC", "I-LOC", "O"],
    },
    {
        "id": "ner_19",
        "text": "جامعة هارفارد تقع في Cambridge بولاية Massachusetts.",
        "tokens": ["جامعة", "هارفارد", "تقع", "في", "Cambridge", "بولاية", "Massachusetts", "."],
        "tags":   ["B-ORG", "I-ORG", "O", "O", "B-LOC", "O", "I-LOC", "O"],
    },
    {
        "id": "ner_20",
        "text": "WHO announced new health guidelines yesterday.",
        "tokens": ["WHO", "announced", "new", "health", "guidelines", "yesterday", "."],
        "tags":   ["B-ORG", "O", "O", "O", "O", "O", "O"],
    },
    # ── Location names ───────────────────────────────────────────────────
    {
        "id": "ner_21",
        "text": "I travelled from London to Riyadh last week.",
        "tokens": ["I", "travelled", "from", "London", "to", "Riyadh", "last", "week", "."],
        "tags":   ["O", "O", "O", "B-LOC", "O", "B-LOC", "O", "O", "O"],
    },
    {
        "id": "ner_22",
        "text": "سافرت من لندن إلى الرياض الأسبوع الماضي.",
        "tokens": ["سافرت", "من", "لندن", "إلى", "الرياض", "الأسبوع", "الماضي", "."],
        "tags":   ["O", "O", "B-LOC", "O", "B-LOC", "O", "O", "O"],
    },
    {
        "id": "ner_23",
        "text": "The Nile River flows through Egypt and Sudan.",
        "tokens": ["The", "Nile", "River", "flows", "through", "Egypt", "and", "Sudan", "."],
        "tags":   ["O", "B-LOC", "I-LOC", "O", "O", "B-LOC", "O", "B-LOC", "O"],
    },
    {
        "id": "ner_24",
        "text": "نهر النيل يتدفق عبر مصر والسودان.",
        "tokens": ["نهر", "النيل", "يتدفق", "عبر", "مصر", "والسودان", "."],
        "tags":   ["B-LOC", "I-LOC", "O", "O", "B-LOC", "B-LOC", "O"],
    },
    {
        "id": "ner_25",
        "text": "Ali moved from Casablanca to Berlin for work.",
        "tokens": ["Ali", "moved", "from", "Casablanca", "to", "Berlin", "for", "work", "."],
        "tags":   ["B-PER", "O", "O", "B-LOC", "O", "B-LOC", "O", "O", "O"],
    },
    # ── Mixed / multi-entity ─────────────────────────────────────────────
    {
        "id": "ner_26",
        "text": "Elon Musk CEO of Tesla announced plans for Mars.",
        "tokens": ["Elon", "Musk", "CEO", "of", "Tesla", "announced", "plans", "for", "Mars", "."],
        "tags":   ["B-PER", "I-PER", "O", "O", "B-ORG", "O", "O", "O", "B-LOC", "O"],
    },
    {
        "id": "ner_27",
        "text": "إيلون ماسك الرئيس التنفيذي لـ Tesla أعلن خططاً للمريخ.",
        "tokens": ["إيلون", "ماسك", "الرئيس", "التنفيذي", "لـ", "Tesla", "أعلن", "خططاً", "للمريخ", "."],
        "tags":   ["B-PER", "I-PER", "O", "O", "O", "B-ORG", "O", "O", "B-LOC", "O"],
    },
    {
        "id": "ner_28",
        "text": "مريم من الإسكندرية تعمل في Google بلندن.",
        "tokens": ["مريم", "من", "الإسكندرية", "تعمل", "في", "Google", "بلندن", "."],
        "tags":   ["B-PER", "O", "B-LOC", "O", "O", "B-ORG", "B-LOC", "O"],
    },
    {
        "id": "ner_29",
        "text": "The FIFA World Cup was held in Qatar in 2022.",
        "tokens": ["The", "FIFA", "World", "Cup", "was", "held", "in", "Qatar", "in", "2022", "."],
        "tags":   ["O", "B-ORG", "I-ORG", "I-ORG", "O", "O", "O", "B-LOC", "O", "O", "O"],
    },
    {
        "id": "ner_30",
        "text": "أُقيم كأس العالم FIFA في قطر عام 2022.",
        "tokens": ["أُقيم", "كأس", "العالم", "FIFA", "في", "قطر", "عام", "2022", "."],
        "tags":   ["O", "B-ORG", "I-ORG", "I-ORG", "O", "B-LOC", "O", "O", "O"],
    },
]


# ---------------------------------------------------------------------------
# Validation + write
# ---------------------------------------------------------------------------

def _validate(examples: list[dict]) -> None:
    for ex in examples:
        assert len(ex["tokens"]) == len(ex["tags"]), (
            f"[{ex['id']}] len(tokens)={len(ex['tokens'])} != len(tags)={len(ex['tags'])}\n"
            f"  tokens: {ex['tokens']}\n"
            f"  tags:   {ex['tags']}"
        )
        valid_tags = {"O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"}
        bad = [t for t in ex["tags"] if t not in valid_tags]
        assert not bad, f"[{ex['id']}] unknown tags: {bad}"


def main() -> None:
    _validate(_EXAMPLES)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        for example in _EXAMPLES:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    # Summary
    tag_counts: dict[str, int] = {}
    for ex in _EXAMPLES:
        for tag in ex["tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print(f"Wrote {len(_EXAMPLES)} examples to {OUTPUT_PATH}")
    print(f"  All len(tokens)==len(tags) checks passed.")
    print("  Tag distribution:")
    for tag in ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]:
        print(f"    {tag:6s}: {tag_counts.get(tag, 0)}")


if __name__ == "__main__":
    main()
