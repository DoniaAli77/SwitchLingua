"""scripts/generate_dummy_data.py

Generates a small Arabic-English code-switched dummy dataset for pipeline
testing and writes it to data/dev_dummy.jsonl (30 examples, 9 topic labels).

Usage:
    python scripts/generate_dummy_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Hard-coded examples (label → list of sentences)
# Each sentence mixes Arabic and English words naturally.
# ---------------------------------------------------------------------------

EXAMPLES: list[dict] = [
    # business (4)
    {"text": "الشركة launched a new product في السوق العالمي.", "label": "business"},
    {"text": "نحن نبحث عن investment opportunities في قطاع التقنية.", "label": "business"},
    {"text": "CEO الشركة أعلن عن merger مع منافسهم الرئيسي.", "label": "business"},
    {"text": "quarterly report أظهر profits غير متوقعة هذا الربع.", "label": "business"},

    # education (3)
    {"text": "الطلاب يستخدمون online courses لتعلم programming.", "label": "education"},
    {"text": "المدرسة أطلقت new curriculum يشمل artificial intelligence.", "label": "education"},
    {"text": "حصلت على scholarship لإكمال master's degree في الخارج.", "label": "education"},

    # health (3)
    {"text": "الأطباء ينصحون بممارسة exercise يومياً للحفاظ على fitness.", "label": "health"},
    {"text": "نظام diet متوازن مهم للحفاظ على healthy lifestyle.", "label": "health"},
    {"text": "تناول vitamins يومياً يحسن من immune system بشكل كبير.", "label": "health"},

    # shopping (3)
    {"text": "اشتريت jacket جديد من online store بسعر مناسب.", "label": "shopping"},
    {"text": "تخفيضات Black Friday في المتاجر الإلكترونية رائعة.", "label": "shopping"},
    {"text": "البضاعة وصلت بعد يومين، delivery كانت سريعة جداً.", "label": "shopping"},

    # medical (3)
    {"text": "الطبيب وصف لي medication جديد لعلاج الضغط.", "label": "medical"},
    {"text": "نتائج scan أظهرت أنه بحاجة إلى surgery بسيطة.", "label": "medical"},
    {"text": "المريض يحتاج إلى follow-up appointment الأسبوع القادم.", "label": "medical"},

    # sports (3)
    {"text": "فريقنا فاز في championship بعد تدريب شاق لأشهر.", "label": "sports"},
    {"text": "اللاعب أحرز hat-trick في آخر مباراة هذا الموسم.", "label": "sports"},
    {"text": "تدريب gym يومياً يحسن من performance في الملعب.", "label": "sports"},

    # tech (4)
    {"text": "الشركة أطلقت smartphone جديد بمواصفات قوية جداً.", "label": "tech"},
    {"text": "تحديث software الأخير أصلح كثيراً من bugs المزعجة.", "label": "tech"},
    {"text": "نظام cloud computing يوفر storage غير محدود للمستخدمين.", "label": "tech"},
    {"text": "أجهزة AI الجديدة تستخدم machine learning للتعرف على الوجوه.", "label": "tech"},

    # finance (3)
    {"text": "سعر الـ dollar ارتفع مقابل العملات الأخرى اليوم.", "label": "finance"},
    {"text": "المستثمرون قلقون من inflation وتأثيرها على portfolio.", "label": "finance"},
    {"text": "البنك يقدم loan بفائدة منخفضة لأصحاب المشاريع الصغيرة.", "label": "finance"},

    # social (4)
    {"text": "نشرت post جديد على Instagram حصل على آلاف likes.", "label": "social"},
    {"text": "trending topic على Twitter كان عن الأحداث المحلية.", "label": "social"},
    {"text": "group على WhatsApp يجمع أصدقاء الجامعة من كل مكان.", "label": "social"},
    {"text": "حضرت online meetup مع community من مختلف الدول العربية.", "label": "social"},
]

assert len(EXAMPLES) == 30, f"Expected 30 examples, got {len(EXAMPLES)}"


def main() -> None:
    output_path = Path(__file__).parent.parent / "data" / "dev_dummy.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        for idx, item in enumerate(EXAMPLES, start=1):
            record = {"id": str(idx), "text": item["text"], "label": item["label"]}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(EXAMPLES)} examples to {output_path}")


if __name__ == "__main__":
    main()
