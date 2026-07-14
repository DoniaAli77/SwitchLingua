"""
profile_generated_corpora.py — intrinsic corpus profile for the generated CS sentiment datasets
================================================================================================
Computes an ArEnTC-style profile for GEN-240 / GEN-480 / GEN-960 (and any extra name=path args):

  1 total examples · 2 class distribution · 3 total tokens · 4-6 sentence length mean/median/p90
  7-9 Arabic / Latin / other token counts + % of all tokens · 10-11 mean per-sentence EN / AR ratio
  12-13 switch points per sentence (mean/median) · 14-15 CMI (mean/median)
  16 % valid code-switched · 17 duplicate count · 18 quality min/max/mean (if column present)

Definitions
-----------
* Tokens = whitespace-split. Per-token script classification by letter counts:
  Arabic letters (U+0600-06FF, U+0750-077F, U+08A0-08FF) vs Latin letters (A-Za-z);
  a token with both is assigned to the majority script; a token with neither is 'other'.
* Switch point = adjacent AR<->EN transition in the token language sequence AFTER dropping 'other'.
* CMI = 100 * (1 - max(ar, en) / (ar + en)); language tokens only; 0 if a sentence has none.
* Valid code-switched = sentence has >=1 Arabic AND >=1 Latin token (same rule as the pipeline's
  deterministic `compute_true_cs_stats.is_code_switched`).
* Duplicates = extra occurrences of a normalized (whitespace-collapsed, lowercased) text.

Outputs (to multi-agent-bert/data/Sentiment/generated/merged/corpus_profile/):
  corpus_profile_per_sentence.csv   (detailed, one row per sentence, all datasets)
  corpus_profile_summary.csv        (one row per dataset x metric-columns)
  corpus_profile_table.tex          (LaTeX table-ready summary: metrics as rows)

Usage:
  python experiments/switchlingua/profile_generated_corpora.py
  python experiments/switchlingua/profile_generated_corpora.py V1_lowerCS=path/to/file.jsonl
"""
import csv
import json
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MERGED = ROOT / "multi-agent-bert" / "data" / "Sentiment" / "generated" / "merged"
OUT = MERGED / "corpus_profile"

DATASETS = {  # name -> jsonl path (order = column order in the LaTeX table)
    "GEN-240": MERGED / "switchlingua_sentiment_train_240_80perlabel.jsonl",
    "GEN-480": MERGED / "switchlingua_sentiment_train_480_160perlabel.jsonl",
    "GEN-960": MERGED / "switchlingua_sentiment_train_960_320perlabel.jsonl",
}

AR_LETTER = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
EN_LETTER = re.compile(r"[A-Za-z]")
LABELS = ("positive", "negative", "neutral")


def classify_token(tok: str) -> str:
    """'ar' | 'en' | 'other' by letter counts; mixed token -> majority script."""
    n_ar = len(AR_LETTER.findall(tok))
    n_en = len(EN_LETTER.findall(tok))
    if n_ar == 0 and n_en == 0:
        return "other"
    if n_ar >= n_en:
        return "ar" if n_ar > 0 else "other"
    return "en"


def profile_sentence(text: str) -> dict:
    tokens = text.split()
    kinds = [classify_token(t) for t in tokens]
    ar = kinds.count("ar")
    en = kinds.count("en")
    other = kinds.count("other")
    lang_seq = [k for k in kinds if k != "other"]
    switches = sum(1 for a, b in zip(lang_seq, lang_seq[1:]) if a != b)
    lang_total = ar + en
    cmi = 100.0 * (1.0 - max(ar, en) / lang_total) if lang_total else 0.0
    return {
        "n_tokens": len(tokens), "ar_tokens": ar, "en_tokens": en, "other_tokens": other,
        "en_ratio": en / lang_total if lang_total else 0.0,
        "ar_ratio": ar / lang_total if lang_total else 0.0,
        "switch_points": switches, "cmi": cmi,
        "is_cs": ar > 0 and en > 0,
    }


def p90(xs):
    return sorted(xs)[int(0.9 * (len(xs) - 1))] if xs else 0


def norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip()).lower()


def load_rows(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def profile_dataset(name: str, path: pathlib.Path):
    rows = load_rows(path)
    per_sent = []
    for i, r in enumerate(rows):
        text = str(r.get("text", ""))
        p = profile_sentence(text)
        q = r.get("quality_score")
        try:
            q = float(q) if q not in (None, "") else None
        except (TypeError, ValueError):
            q = None
        per_sent.append({"dataset": name, "idx": i, "label": r.get("label", ""),
                         "quality_score": q, **p, "text": text})

    n = len(per_sent)
    lens = [s["n_tokens"] for s in per_sent]
    tot_tokens = sum(lens)
    tot_ar = sum(s["ar_tokens"] for s in per_sent)
    tot_en = sum(s["en_tokens"] for s in per_sent)
    tot_other = sum(s["other_tokens"] for s in per_sent)
    cmis = [s["cmi"] for s in per_sent]
    sw = [s["switch_points"] for s in per_sent]
    dup = n - len({norm_text(s["text"]) for s in per_sent})
    quality = [s["quality_score"] for s in per_sent if s["quality_score"] is not None]
    class_dist = {lab: sum(1 for s in per_sent if s["label"] == lab) for lab in LABELS}

    summary = {
        "dataset": name,
        "total_examples": n,
        **{f"n_{lab}": class_dist[lab] for lab in LABELS},
        "total_tokens": tot_tokens,
        "len_mean": round(statistics.mean(lens), 2),
        "len_median": statistics.median(lens),
        "len_p90": p90(lens),
        "ar_tokens": tot_ar, "ar_token_pct": round(100 * tot_ar / tot_tokens, 2),
        "en_tokens": tot_en, "en_token_pct": round(100 * tot_en / tot_tokens, 2),
        "other_tokens": tot_other, "other_token_pct": round(100 * tot_other / tot_tokens, 2),
        "mean_en_ratio": round(statistics.mean(s["en_ratio"] for s in per_sent), 4),
        "mean_ar_ratio": round(statistics.mean(s["ar_ratio"] for s in per_sent), 4),
        "switch_points_mean": round(statistics.mean(sw), 2),
        "switch_points_median": statistics.median(sw),
        "cmi_mean": round(statistics.mean(cmis), 2),
        "cmi_median": round(statistics.median(cmis), 2),
        "cs_valid_pct": round(100 * sum(s["is_cs"] for s in per_sent) / n, 2),
        "duplicates": dup,
        "quality_min": round(min(quality), 2) if quality else "",
        "quality_max": round(max(quality), 2) if quality else "",
        "quality_mean": round(statistics.mean(quality), 3) if quality else "",
    }
    return summary, per_sent


LATEX_ROWS = [  # (metric label, summary key, format)
    ("Total examples", "total_examples", "{:d}"),
    ("~~positive / negative / neutral", None, None),  # composed
    ("Total tokens", "total_tokens", "{:,d}"),
    ("Sentence length (mean)", "len_mean", "{:.2f}"),
    ("Sentence length (median)", "len_median", "{:.0f}"),
    ("Sentence length (P90)", "len_p90", "{:.0f}"),
    ("Arabic tokens (\\%)", None, None),   # composed count + pct
    ("English tokens (\\%)", None, None),
    ("Other tokens (\\%)", None, None),
    ("Mean Arabic ratio", "mean_ar_ratio", "{:.3f}"),
    ("Mean English ratio", "mean_en_ratio", "{:.3f}"),
    ("Switch points / sentence (mean)", "switch_points_mean", "{:.2f}"),
    ("Switch points / sentence (median)", "switch_points_median", "{:.0f}"),
    ("CMI (mean)", "cmi_mean", "{:.2f}"),
    ("CMI (median)", "cmi_median", "{:.2f}"),
    ("Valid code-switched (\\%)", "cs_valid_pct", "{:.1f}"),
    ("Duplicates", "duplicates", "{:d}"),
    ("Quality score (min/mean/max)", None, None),
]


def latex_table(summaries: list[dict]) -> str:
    names = [s["dataset"] for s in summaries]
    lines = [
        "% Auto-generated by profile_generated_corpora.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Intrinsic corpus profile of the generated Arabic--English code-switched sentiment datasets.}",
        "\\label{tab:gen-corpus-profile}",
        "\\begin{tabular}{l" + "r" * len(names) + "}",
        "\\toprule",
        "Metric & " + " & ".join(names) + " \\\\",
        "\\midrule",
    ]
    for label, key, fmt in LATEX_ROWS:
        cells = []
        for s in summaries:
            if key is not None:
                v = s[key]
                cells.append(fmt.format(v) if v != "" else "--")
            elif label.startswith("~~positive"):
                cells.append(f"{s['n_positive']} / {s['n_negative']} / {s['n_neutral']}")
            elif label.startswith("Arabic tokens"):
                cells.append(f"{s['ar_tokens']:,} ({s['ar_token_pct']:.1f})")
            elif label.startswith("English tokens"):
                cells.append(f"{s['en_tokens']:,} ({s['en_token_pct']:.1f})")
            elif label.startswith("Other tokens"):
                cells.append(f"{s['other_tokens']:,} ({s['other_token_pct']:.1f})")
            elif label.startswith("Quality"):
                cells.append(f"{s['quality_min']} / {s['quality_mean']} / {s['quality_max']}"
                             if s["quality_mean"] != "" else "--")
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main():
    datasets = dict(DATASETS)
    for arg in sys.argv[1:]:  # optional extras: name=path
        if "=" in arg:
            k, v = arg.split("=", 1)
            datasets[k] = pathlib.Path(v)

    OUT.mkdir(parents=True, exist_ok=True)
    summaries, all_sent = [], []
    for name, path in datasets.items():
        if not path.exists():
            print(f"SKIP {name}: {path} not found")
            continue
        summary, per_sent = profile_dataset(name, path)
        summaries.append(summary)
        all_sent.extend(per_sent)
        print(f"[{name}] n={summary['total_examples']} tokens={summary['total_tokens']:,} "
              f"AR%={summary['ar_token_pct']} EN%={summary['en_token_pct']} "
              f"CMI={summary['cmi_mean']} switches={summary['switch_points_mean']} "
              f"CS-valid={summary['cs_valid_pct']}% dup={summary['duplicates']}")

    # detailed per-sentence CSV
    sent_cols = ["dataset", "idx", "label", "n_tokens", "ar_tokens", "en_tokens", "other_tokens",
                 "ar_ratio", "en_ratio", "switch_points", "cmi", "is_cs", "quality_score", "text"]
    with open(OUT / "corpus_profile_per_sentence.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=sent_cols, extrasaction="ignore")
        w.writeheader()
        for s in all_sent:
            s = dict(s)
            s["ar_ratio"] = round(s["ar_ratio"], 4)
            s["en_ratio"] = round(s["en_ratio"], 4)
            s["cmi"] = round(s["cmi"], 2)
            w.writerow(s)

    # summary CSV (datasets as rows)
    if summaries:
        with open(OUT / "corpus_profile_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)

    # LaTeX table
    (OUT / "corpus_profile_table.tex").write_text(latex_table(summaries), encoding="utf-8")
    print(f"\nwrote {OUT / 'corpus_profile_per_sentence.csv'}")
    print(f"wrote {OUT / 'corpus_profile_summary.csv'}")
    print(f"wrote {OUT / 'corpus_profile_table.tex'}")


if __name__ == "__main__":
    main()
