"""
build_csratio_set.py — assemble the fixed 30-sentence CS-ratio validation set.
20 real sentences (sampled from the consolidated annotation sheet, mixed tasks/conditions)
+ 10 controlled edge cases (measurement stress test). Human count columns left blank.
Output: experiments/outputs/switchlingua/csratio/csratio_validation_set.csv
"""
import csv, pathlib, random

ROOT = pathlib.Path(__file__).resolve().parents[2]
SHEET = ROOT / "experiments" / "outputs" / "switchlingua" / "human_eval" / "consolidated_human_annotation_sheet.csv"
OUT = ROOT / "experiments" / "outputs" / "switchlingua" / "csratio"
OUT.mkdir(parents=True, exist_ok=True)

COLS = ["sample_id", "source", "text", "is_controlled_edge_case", "edge_case_type",
        "human_arabic_token_count", "human_english_token_count", "human_other_token_count", "human_notes"]

EDGE_CASES = [
    ("fully_arabic", "أنا أحب القراءة في المكتبة كل صباح قبل الذهاب إلى العمل."),
    ("fully_english", "I really enjoy reading books at the library every morning before work."),
    ("mostly_arabic_one_english_word", "ذهبت إلى الـ mall أمس واشتريت بعض الملابس الجديدة لأصدقائي."),
    ("mostly_english_one_arabic_word", "I went to the مطعم yesterday and the food there was absolutely delicious."),
    ("balanced_arabic_english", "اليوم كان يوماً رائعاً جداً, and tomorrow is going to be even better than this."),
    ("arabic_with_english_person", "تحدثت اليوم مع Sarah Hassan حول تفاصيل المشروع الجديد في الشركة."),
    ("arabic_with_english_product", "اشتريت جهاز iPhone الجديد من المتجر وهو سريع جداً في الأداء."),
    ("arabic_with_english_event", "أنا متحمس جداً لحضور World Cup هذا العام مع عائلتي وأصدقائي."),
    ("english_with_arabic_phrase", "The restaurant was excellent and honestly كان السعر مناسب جداً for the quality."),
    ("punctuation_numbers_mixed", "في عام 2024، اشترينا 3 أجهزة laptops بسعر $1500 لكل واحد!!! :)"),
]


def main():
    rows = []
    # --- 20 real sentences, mixed across task + masked/control ---
    if SHEET.exists():
        sheet = list(csv.DictReader(SHEET.open(encoding="utf-8-sig")))
        random.seed(11)
        random.shuffle(sheet)
        quota = {("topic", "any"): 4, ("sentiment", "any"): 5, ("ner", "any"): 4,
                 ("any", "masked"): 4, ("any", "control"): 3}
        got = {k: 0 for k in quota}
        used = set()

        def take(pred, key):
            for r in sheet:
                if r["text"] in used:
                    continue
                if pred(r) and got[key] < quota[key]:
                    used.add(r["text"]); got[key] += 1
                    rows.append({
                        "source": f"modified/{r['task']}" + ("/masked" if r.get("masked_case") == "yes" else ""),
                        "text": r["text"], "is_controlled_edge_case": "no", "edge_case_type": "",
                    })
        take(lambda r: r["task"] == "topic", ("topic", "any"))
        take(lambda r: r["task"] == "sentiment", ("sentiment", "any"))
        take(lambda r: r["task"] == "ner", ("ner", "any"))
        take(lambda r: r.get("masked_case") == "yes", ("any", "masked"))
        take(lambda r: r.get("masked_case") == "no", ("any", "control"))
        # top up to 20 from whatever remains
        for r in sheet:
            if len(rows) >= 20:
                break
            if r["text"] not in used:
                used.add(r["text"])
                rows.append({"source": f"modified/{r['task']}", "text": r["text"],
                             "is_controlled_edge_case": "no", "edge_case_type": ""})
    else:
        print(f"WARNING: {SHEET} not found - only edge cases will be written")

    # --- 10 controlled edge cases ---
    for etype, text in EDGE_CASES:
        rows.append({"source": "controlled_edge", "text": text,
                     "is_controlled_edge_case": "yes", "edge_case_type": etype})

    with open(OUT / "csratio_validation_set.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for i, r in enumerate(rows, 1):
            r["sample_id"] = f"CS{i:03d}"
            for c in ("human_arabic_token_count", "human_english_token_count", "human_other_token_count", "human_notes"):
                r.setdefault(c, "")
            w.writerow({c: r.get(c, "") for c in COLS})

    n_real = sum(1 for r in rows if r["is_controlled_edge_case"] == "no")
    print(f"Wrote {len(rows)} sentences ({n_real} real + {len(rows)-n_real} edge cases) -> {OUT/'csratio_validation_set.csv'}")


if __name__ == "__main__":
    main()
