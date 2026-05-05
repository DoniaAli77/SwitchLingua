"""scripts/generate_dummy_sentiment_data.py

Generates ``data/dev_dummy_sentiment.jsonl`` — 30 Arabic-English code-switched
examples for the sentiment_classification task:
  • 10 positive
  • 10 negative
  • 10 neutral

Usage
-----
    python scripts/generate_dummy_sentiment_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "dev_dummy_sentiment.jsonl"

_EXAMPLES: list[dict[str, str]] = [
    # ── positive (10) ─────────────────────────────────────────────────────
    {
        "id":    "sent_01",
        "label": "positive",
        "text":  "The new phone is great, شاشتها رائعة و الأداء ممتاز بالفعل.",
    },
    {
        "id":    "sent_02",
        "label": "positive",
        "text":  "أنا مبسوط جداً بالخدمة, the staff were wonderful and very helpful.",
    },
    {
        "id":    "sent_03",
        "label": "positive",
        "text":  "I love this restaurant, الأكل لذيذ والجو fantastic.",
    },
    {
        "id":    "sent_04",
        "label": "positive",
        "text":  "الفيلم كان amazing, قصة excellent وممثلين رائعين.",
    },
    {
        "id":    "sent_05",
        "label": "positive",
        "text":  "Best experience ever, الخدمة كانت ممتازة وسريعة جداً.",
    },
    {
        "id":    "sent_06",
        "label": "positive",
        "text":  "هذا المنتج wonderful, I would highly recommend it to everyone.",
    },
    {
        "id":    "sent_07",
        "label": "positive",
        "text":  "The class was fantastic, الأستاذ شرح بطريقة جيدة وواضحة.",
    },
    {
        "id":    "sent_08",
        "label": "positive",
        "text":  "سعيد جداً بالنتيجة, the team did an amazing job this season.",
    },
    {
        "id":    "sent_09",
        "label": "positive",
        "text":  "القهوة كانت great, I will definitely come back again.",
    },
    {
        "id":    "sent_10",
        "label": "positive",
        "text":  "Excellent update, التطبيق أصبح أسرع بكثير وأكثر استقراراً.",
    },
    # ── negative (10) ─────────────────────────────────────────────────────
    {
        "id":    "sent_11",
        "label": "negative",
        "text":  "The delivery was terrible, الطلب وصل متأخر جداً وكان مكسور.",
    },
    {
        "id":    "sent_12",
        "label": "negative",
        "text":  "أنا زعلان من الخدمة, the staff were awful and very rude.",
    },
    {
        "id":    "sent_13",
        "label": "negative",
        "text":  "I hate this app, يتوقف دايماً ولا يعمل بشكل صحيح أبداً.",
    },
    {
        "id":    "sent_14",
        "label": "negative",
        "text":  "الطعام كان horrible, worst meal I have had in a long time.",
    },
    {
        "id":    "sent_15",
        "label": "negative",
        "text":  "Very bad customer service, ما ردوا علي ولم يحلوا المشكلة.",
    },
    {
        "id":    "sent_16",
        "label": "negative",
        "text":  "هذا المنتج رهيب, the quality is so poor it broke after one day.",
    },
    {
        "id":    "sent_17",
        "label": "negative",
        "text":  "The movie was disgusting, قصة سيئة وممثلين بدون موهبة.",
    },
    {
        "id":    "sent_18",
        "label": "negative",
        "text":  "مروع جداً هذا التطبيق, it crashes every time I open it.",
    },
    {
        "id":    "sent_19",
        "label": "negative",
        "text":  "Worst hotel ever, الغرفة كانت بشعة وغير نظيفة للأسف.",
    },
    {
        "id":    "sent_20",
        "label": "negative",
        "text":  "لا أنصح بهذا المطعم أبداً, the food was awful and overpriced.",
    },
    # ── neutral (10) ──────────────────────────────────────────────────────
    {
        "id":    "sent_21",
        "label": "neutral",
        "text":  "The meeting is scheduled for Tuesday, الاجتماع يوم الثلاثاء الساعة 10.",
    },
    {
        "id":    "sent_22",
        "label": "neutral",
        "text":  "الطقس اليوم moderate, temperatures are around 25 degrees.",
    },
    {
        "id":    "sent_23",
        "label": "neutral",
        "text":  "The report includes three sections, التقرير يحتوي على ثلاثة أقسام.",
    },
    {
        "id":    "sent_24",
        "label": "neutral",
        "text":  "المتجر يفتح الساعة التاسعة, it closes at nine in the evening.",
    },
    {
        "id":    "sent_25",
        "label": "neutral",
        "text":  "The package weighs two kilograms, الطرد وزنه كيلوغرامان تقريباً.",
    },
    {
        "id":    "sent_26",
        "label": "neutral",
        "text":  "قرأت المقالة كاملاً, the article covered an average of ten topics.",
    },
    {
        "id":    "sent_27",
        "label": "neutral",
        "text":  "The bus arrives every 30 minutes, الحافلة تأتي كل نصف ساعة.",
    },
    {
        "id":    "sent_28",
        "label": "neutral",
        "text":  "هذه المعلومات عادية ومتاحة, the data is publicly available online.",
    },
    {
        "id":    "sent_29",
        "label": "neutral",
        "text":  "The update is a normal maintenance release, تحديث صيانة اعتيادي.",
    },
    {
        "id":    "sent_30",
        "label": "neutral",
        "text":  "الموعد fine, we can reschedule if needed without any problem.",
    },
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        for example in _EXAMPLES:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    label_counts: dict[str, int] = {}
    for ex in _EXAMPLES:
        label_counts[ex["label"]] = label_counts.get(ex["label"], 0) + 1

    print(f"Wrote {len(_EXAMPLES)} examples to {OUTPUT_PATH}")
    for lbl, count in sorted(label_counts.items()):
        print(f"  {lbl:10s}: {count}")


if __name__ == "__main__":
    main()
