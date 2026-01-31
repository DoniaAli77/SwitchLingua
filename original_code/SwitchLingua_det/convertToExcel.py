import json
import pandas as pd

INPUT_JSONL = "./output/Arabic.jsonl"   # adjust if needed
OUTPUT_EXCEL = "./output/Arabic_analysis.xlsx"

def join_any(lst):
    """Join list items into one string; supports str/dict/etc."""
    if not lst:
        return ""
    out = []
    for x in lst:
        if isinstance(x, str):
            out.append(x)
        else:
            out.append(json.dumps(x, ensure_ascii=False))
    return "; ".join(out)

rows = []

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        obj = json.loads(line)

        # -------- scenario metadata --------
        meta = {k: obj.get(k) for k in [
            "topic","tense","perspective","cs_ratio","gender","age","education_level",
            "first_language","second_language","conversation_type","cs_function","cs_type",
            "score","refine_count"
        ]}

        # -------- nested agent results (your real structure) --------
        flu = obj.get("fluency_result") or {}
        nat = obj.get("naturalness_result") or {}
        csr = obj.get("cs_ratio_result") or {}
        soc = obj.get("social_cultural_result") or {}

        agent = {
            # scores
            "fluency_score": flu.get("fluency_score"),
            "naturalness_score": nat.get("naturalness_score"),
            "cs_ratio_score": csr.get("ratio_score"),
            "socio_cultural_score": soc.get("socio_cultural_score"),

            # details you asked for
            "fluency_errors": join_any(flu.get("errors")),
            "naturalness_observations": join_any(nat.get("observations")),
            "cs_ratio_computed": csr.get("computed_ratio"),
            "cs_ratio_notes": csr.get("notes"),
            "socio_cultural_issues": join_any(soc.get("issues")),

            # (optional) summaries (nice for analysis)
            "fluency_summary": flu.get("summary"),
            "naturalness_summary": nat.get("summary"),
            "socio_cultural_summary": soc.get("summary"),
        }

        # -------- generated sentences: list -> one row per sentence --------
        sents = obj.get("data_generation_result")
        if isinstance(sents, list) and len(sents) > 0:
            for i, sent in enumerate(sents):
                rows.append({**meta, **agent, "sentence_index": i, "text": sent})
        else:
            rows.append({**meta, **agent, "sentence_index": 0, "text": sents})

df = pd.DataFrame(rows)

# nicer column order
preferred = [
    "topic","text","score",
    "fluency_score","naturalness_score","cs_ratio_score","socio_cultural_score",
    "fluency_errors","naturalness_observations","cs_ratio_computed","cs_ratio_notes","socio_cultural_issues",
    "tense","perspective","conversation_type","cs_ratio","cs_type","cs_function",
    "gender","age","education_level","refine_count","sentence_index",
    "fluency_summary","naturalness_summary","socio_cultural_summary"
]
ordered = [c for c in preferred if c in df.columns]
df = df[ordered + [c for c in df.columns if c not in ordered]]

df.to_excel(OUTPUT_EXCEL, index=False)
print(f"✅ Saved {len(df)} rows to {OUTPUT_EXCEL}")



import json
import pandas as pd
from pathlib import Path

# -------- paths --------
INPUT_JSONL = "./output/Arabic.jsonl"   # adjust if needed
OUTPUT_EXCEL = "./output/Arabic.xlsx"

rows = []

# -------- read jsonl --------
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            rows.append(obj)

# -------- convert to DataFrame --------
df = pd.DataFrame(rows)

# -------- optional: column ordering --------
preferred_cols = [
    "topic",
    "text",
    "score",
    "cs_ratio",
    "cs_type",
    "cs_function",
    "tense",
    "perspective",
    "conversation_type",
    "first_language",
    "second_language",
    "gender",
    "age",
    "education_level",
    "refine_count",
]

# keep only existing columns (safe)
ordered_cols = [c for c in preferred_cols if c in df.columns]
df = df[ordered_cols + [c for c in df.columns if c not in ordered_cols]]

# -------- save to Excel --------
df.to_excel(OUTPUT_EXCEL, index=False)

print(f"✅ Saved {len(df)} rows to {OUTPUT_EXCEL}")