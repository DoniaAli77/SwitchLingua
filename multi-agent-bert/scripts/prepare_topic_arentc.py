"""Inspect, validate, normalize, and export the ARENTC topic datasets (V1 & V2).

Inspection/prep only — no fine-tuning. Outputs JSONL + a stats summary (JSON to
stdout) used to build the readiness report. Does not touch the original .xlsx.
"""
from __future__ import annotations
import json, re, os, sys
import pandas as pd

LABELS = ["business","education","finance","health","medical","shopping","social","sports","tech"]
LABELSET = set(LABELS)

VERSIONS = {
    "ARENTCV1": {
        "train": "data/Topic/ARENTCV1/train_split.xlsx",
        "dev":   "data/Topic/ARENTCV1/dev_split.xlsx",
        "test":  "data/Topic/ARENTCV1/test_split.xlsx",
    },
    "ARENTCV2": {
        "train": "data/Topic/ARENTCV2/train_split_no_fully_arabic.xlsx",
        "dev":   "data/Topic/ARENTCV2/dev_split_no_fully_arabic.xlsx",
        "test":  "data/Topic/ARENTCV2/test_split_no_fully_arabic.xlsx",
    },
}

ARAB = re.compile(r"[؀-ۿ]")
LAT  = re.compile(r"[A-Za-z]")

def cs_status(t: str) -> str:
    a, e = bool(ARAB.search(t)), bool(LAT.search(t))
    if a and e: return "code_switched"
    if a: return "fully_arabic"
    if e: return "fully_english"
    return "other"

def norm_text(t: str) -> str:
    return " ".join(str(t).split()).lower()

summary = {}
for ver, files in VERSIONS.items():
    outdir = f"data/Topic/processed/{ver}"
    os.makedirs(outdir, exist_ok=True)
    vstat = {"files": files, "splits": {}, "label_set_ok": True, "unexpected_labels": []}
    norm_sets = {}
    for split, path in files.items():
        df = pd.read_excel(path)
        n0 = len(df)
        # normalize labels
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        labs = sorted(df["label"].unique().tolist())
        bad = [l for l in labs if l not in LABELSET]
        if bad:
            vstat["label_set_ok"] = False
            vstat["unexpected_labels"] += [(split, b) for b in bad]
        # empty text
        df["text"] = df["text"].astype(str)
        empty = int((df["text"].str.strip() == "").sum() + df["text"].str.lower().isin(["nan","none"]).sum())
        # dup within split (normalized text)
        nt = df["text"].map(norm_text)
        dup = int(nt.duplicated().sum())
        norm_sets[split] = set(nt)
        # cs status
        cs = df["text"].map(cs_status).value_counts().to_dict()
        # label distribution
        dist = df["label"].value_counts().to_dict()
        # export JSONL
        outp = f"{outdir}/{split}.jsonl"
        with open(outp, "w", encoding="utf-8") as f:
            for i, row in enumerate(df.itertuples(index=False)):
                f.write(json.dumps({"id": f"{ver}-{split}-{i:06d}",
                                    "text": row.text, "label": row.label},
                                   ensure_ascii=False) + "\n")
        vstat["splits"][split] = {
            "rows": n0, "labels": labs, "label_dist": {k:int(v) for k,v in dist.items()},
            "empty_text": empty, "dup_within": dup,
            "cs_status": {k:int(v) for k,v in cs.items()},
            "jsonl": outp,
        }
    # leakage across splits (normalized text overlap)
    def ov(a,b): return len(norm_sets[a] & norm_sets[b])
    vstat["leakage"] = {
        "train_dev": ov("train","dev"), "train_test": ov("train","test"),
        "dev_test": ov("dev","test"),
    }
    summary[ver] = vstat

print(json.dumps(summary, ensure_ascii=False, indent=2))
