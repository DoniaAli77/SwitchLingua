"""profile_ner_corpora.py — intrinsic corpus profile for the generated NER datasets + the real Sabty corpus
=============================================================================================================
Same methodology as profile_generated_corpora.py (the GEN-240/480/960 sentiment profile), extended with
entity-level statistics, applied to:
  * our generated NER ladder: NER-240 / NER-480 / NER-720 / NER-960
  * the REAL Sabty AR-EN NER corpus (Train_AR-EN_NER.txt + Test_AR-EN_NER.txt + combined)

There is no published intrinsic-eval table from the Sabty paper available in this repo to cite numbers
from directly, so instead of quoting unverifiable figures, this script computes the SAME metrics on her
actual released files -- giving a real, reproducible baseline to compare our generated data against
rather than a claim from memory.

Definitions (identical to profile_generated_corpora.py)
---------------------------------------------------------
* Tokens = whitespace-split (generated data) / corpus-provided tokens (Sabty, already pre-tokenized).
  Per-token script: Arabic letters (U+0600-06FF, U+0750-077F, U+08A0-08FF) vs Latin (A-Za-z); a token
  with both is assigned to the majority script; a token with neither is 'other'.
* Switch point = adjacent AR<->EN transition in the token language sequence AFTER dropping 'other'.
* CMI = 100 * (1 - max(ar, en) / (ar + en)); language tokens only; 0 if a sentence has none.
* Valid code-switched = sentence has >=1 Arabic AND >=1 Latin token.

Entity-level additions (NER-specific, not in the sentiment profile)
---------------------------------------------------------------------
* Entity-bearing rate = % of sentences with >=1 entity.
* Entities / sentence = mean count over ALL sentences (0 for entity-free ones).
* Entity type distribution = share of ENTITY-BEARING sentences whose type-set equals each combo
  (e.g. "LOC", "PER+LOC"), i.e. the same "group" accounting used for the generation quotas.
* Entity span length = mean tokens per entity span.
* Sabty entities are recovered from its IO tag scheme (only 'I-TYPE' prefixes, no 'B-'): a run of
  consecutive tokens with the same type is treated as one entity. This under-splits two adjacent
  same-type entities with no separator into a single span -- a known limitation of IO tagging, not
  a bug in this script. Sabty tags persons as PERS; our pipeline's internal tag is PER (same class).

Outputs -> multi-agent-bert/data/NER/generated/expN/merged/corpus_profile/
  ner_corpus_profile_per_sentence.csv
  ner_corpus_profile_summary.csv
  ner_corpus_profile_table.tex
"""
import csv
import json
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "multi-agent-bert" / "src" / "evaluation"))
from ner_conll_loader import load_conll, tag_to_type  # noqa: E402

NER_DIR = ROOT / "multi-agent-bert" / "data" / "NER"
MERGED = NER_DIR / "generated" / "expN" / "merged"
OUT = MERGED / "corpus_profile"

GEN_DATASETS = {
    "NER-240": MERGED / "switchlingua_ner_train_240.jsonl",
    "NER-480": MERGED / "switchlingua_ner_train_480.jsonl",
    "NER-720": MERGED / "switchlingua_ner_train_720.jsonl",
    "NER-960": MERGED / "switchlingua_ner_train_960.jsonl",
}
SABTY_FILES = {
    "Sabty-Train": NER_DIR / "Train_AR-EN_NER.txt",
    "Sabty-Test": NER_DIR / "Test_AR-EN_NER.txt",
}

AR_LETTER = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
EN_LETTER = re.compile(r"[A-Za-z]")


def classify_token(tok: str) -> str:
    n_ar = len(AR_LETTER.findall(tok))
    n_en = len(EN_LETTER.findall(tok))
    if n_ar == 0 and n_en == 0:
        return "other"
    if n_ar >= n_en:
        return "ar" if n_ar > 0 else "other"
    return "en"


def profile_tokens(tokens: list[str]) -> dict:
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


def entities_from_io_tags(tokens: list[str], tags: list[str]) -> list[dict]:
    """Group consecutive same-type IO tags into entity spans (Sabty scheme)."""
    ents, cur_type, cur_toks = [], None, []
    for tok, tag in zip(tokens, tags):
        t = tag_to_type(tag)
        if t == cur_type and t != "O":
            cur_toks.append(tok)
            continue
        if cur_type and cur_type != "O":
            ents.append({"type": cur_type, "n_tokens": len(cur_toks)})
        cur_type, cur_toks = t, [tok] if t != "O" else []
    if cur_type and cur_type != "O":
        ents.append({"type": cur_type, "n_tokens": len(cur_toks)})
    return ents


def load_generated(name: str, path: pathlib.Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        text = str(r.get("text", ""))
        tokens = text.split()
        p = profile_tokens(tokens)
        ents = r.get("entities_verified") or []
        types_present = sorted({e["type"] for e in ents})
        q = r.get("quality_score")
        try:
            q = float(q) if q not in (None, "") else None
        except (TypeError, ValueError):
            q = None
        rows.append({
            "dataset": name, "idx": i, **p, "text": text,
            "n_entities": len(ents),
            "entity_types": "+".join(types_present) if types_present else "",
            "quality_score": q,
        })
    return rows


def load_sabty(name: str, path: pathlib.Path) -> list[dict]:
    sentences = load_conll(path)
    rows = []
    for i, s in enumerate(sentences):
        tokens = s["tokens"]
        if not tokens:
            continue
        p = profile_tokens(tokens)
        ents = entities_from_io_tags(tokens, s["tags"])
        types_present = sorted({e["type"] for e in ents})
        rows.append({
            "dataset": name, "idx": i, **p, "text": " ".join(tokens),
            "n_entities": len(ents),
            "entity_types": "+".join(types_present) if types_present else "",
            "quality_score": None,
        })
    return rows


def profile_dataset(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    lens = [r["n_tokens"] for r in rows]
    tot_tokens = sum(lens)
    tot_ar = sum(r["ar_tokens"] for r in rows)
    tot_en = sum(r["en_tokens"] for r in rows)
    tot_other = sum(r["other_tokens"] for r in rows)
    cmis = [r["cmi"] for r in rows]
    sw = [r["switch_points"] for r in rows]
    dup = n - len({norm_text(r["text"]) for r in rows})
    quality = [r["quality_score"] for r in rows if r["quality_score"] is not None]

    bearing = [r for r in rows if r["n_entities"] > 0]
    n_bearing = len(bearing)
    combo_counts: dict[str, int] = {}
    for r in bearing:
        combo_counts[r["entity_types"]] = combo_counts.get(r["entity_types"], 0) + 1
    combo_share = {k: round(100 * v / n_bearing, 2) for k, v in combo_counts.items()} if n_bearing else {}

    return {
        "dataset": name,
        "total_examples": n,
        "total_tokens": tot_tokens,
        "len_mean": round(statistics.mean(lens), 2) if lens else 0,
        "len_median": statistics.median(lens) if lens else 0,
        "len_p90": p90(lens),
        "ar_tokens": tot_ar, "ar_token_pct": round(100 * tot_ar / tot_tokens, 2) if tot_tokens else 0,
        "en_tokens": tot_en, "en_token_pct": round(100 * tot_en / tot_tokens, 2) if tot_tokens else 0,
        "other_tokens": tot_other, "other_token_pct": round(100 * tot_other / tot_tokens, 2) if tot_tokens else 0,
        "mean_ar_ratio": round(statistics.mean(r["ar_ratio"] for r in rows), 4) if rows else 0,
        "mean_en_ratio": round(statistics.mean(r["en_ratio"] for r in rows), 4) if rows else 0,
        "switch_points_mean": round(statistics.mean(sw), 2) if sw else 0,
        "switch_points_median": statistics.median(sw) if sw else 0,
        "cmi_mean": round(statistics.mean(cmis), 2) if cmis else 0,
        "cmi_median": round(statistics.median(cmis), 2) if cmis else 0,
        "cs_valid_pct": round(100 * sum(r["is_cs"] for r in rows) / n, 2) if n else 0,
        "duplicates": dup,
        "quality_min": round(min(quality), 2) if quality else "",
        "quality_max": round(max(quality), 2) if quality else "",
        "quality_mean": round(statistics.mean(quality), 3) if quality else "",
        "entity_bearing_pct": round(100 * n_bearing / n, 2) if n else 0,
        "entities_per_sentence_mean": round(sum(r["n_entities"] for r in rows) / n, 3) if n else 0,
        "entity_combo_share_pct": combo_share,
    }


LATEX_ROWS = [
    ("Total examples", "total_examples", "{:d}"),
    ("Total tokens", "total_tokens", "{:,d}"),
    ("Sentence length (mean)", "len_mean", "{:.2f}"),
    ("Sentence length (median)", "len_median", "{:.0f}"),
    ("Arabic tokens (\\%)", None, None),
    ("English tokens (\\%)", None, None),
    ("Other tokens (\\%)", None, None),
    ("Mean Arabic ratio", "mean_ar_ratio", "{:.3f}"),
    ("Mean English ratio", "mean_en_ratio", "{:.3f}"),
    ("Switch points / sentence (mean)", "switch_points_mean", "{:.2f}"),
    ("CMI (mean)", "cmi_mean", "{:.2f}"),
    ("CMI (median)", "cmi_median", "{:.2f}"),
    ("Valid code-switched (\\%)", "cs_valid_pct", "{:.1f}"),
    ("Entity-bearing sentences (\\%)", "entity_bearing_pct", "{:.1f}"),
    ("Entities / sentence (mean)", "entities_per_sentence_mean", "{:.3f}"),
    ("Duplicates", "duplicates", "{:d}"),
]


def latex_table(summaries: list[dict]) -> str:
    names = [s["dataset"] for s in summaries]
    lines = [
        "% Auto-generated by profile_ner_corpora.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Intrinsic corpus profile of the generated NER datasets vs. the real Sabty AR--EN NER corpus.}",
        "\\label{tab:ner-corpus-profile}",
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
            elif label.startswith("Arabic tokens"):
                cells.append(f"{s['ar_tokens']:,} ({s['ar_token_pct']:.1f})")
            elif label.startswith("English tokens"):
                cells.append(f"{s['en_tokens']:,} ({s['en_token_pct']:.1f})")
            elif label.startswith("Other tokens"):
                cells.append(f"{s['other_tokens']:,} ({s['other_token_pct']:.1f})")
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, all_rows = [], []

    for name, path in GEN_DATASETS.items():
        if not path.exists():
            print(f"SKIP {name}: {path} not found")
            continue
        rows = load_generated(name, path)
        s = profile_dataset(name, rows)
        summaries.append(s)
        all_rows.extend(rows)
        print(f"[{name}] n={s['total_examples']} tokens={s['total_tokens']:,} "
              f"AR%={s['ar_token_pct']} EN%={s['en_token_pct']} CMI={s['cmi_mean']} "
              f"switches={s['switch_points_mean']} CS-valid={s['cs_valid_pct']}% "
              f"entity-bearing={s['entity_bearing_pct']}% ent/sent={s['entities_per_sentence_mean']} "
              f"dup={s['duplicates']}")

    sabty_rows_all = []
    for name, path in SABTY_FILES.items():
        if not path.exists():
            print(f"SKIP {name}: {path} not found")
            continue
        rows = load_sabty(name, path)
        sabty_rows_all.extend(rows)
        s = profile_dataset(name, rows)
        summaries.append(s)
        all_rows.extend(rows)
        print(f"[{name}] n={s['total_examples']} tokens={s['total_tokens']:,} "
              f"AR%={s['ar_token_pct']} EN%={s['en_token_pct']} CMI={s['cmi_mean']} "
              f"switches={s['switch_points_mean']} CS-valid={s['cs_valid_pct']}% "
              f"entity-bearing={s['entity_bearing_pct']}% ent/sent={s['entities_per_sentence_mean']}")

    if sabty_rows_all:
        s = profile_dataset("Sabty-All", sabty_rows_all)
        summaries.append(s)
        print(f"[Sabty-All] n={s['total_examples']} tokens={s['total_tokens']:,} "
              f"AR%={s['ar_token_pct']} EN%={s['en_token_pct']} CMI={s['cmi_mean']} "
              f"switches={s['switch_points_mean']} CS-valid={s['cs_valid_pct']}% "
              f"entity-bearing={s['entity_bearing_pct']}% ent/sent={s['entities_per_sentence_mean']}")

    # per-sentence CSV
    cols = ["dataset", "idx", "n_tokens", "ar_tokens", "en_tokens", "other_tokens",
            "ar_ratio", "en_ratio", "switch_points", "cmi", "is_cs",
            "n_entities", "entity_types", "quality_score", "text"]
    with open(OUT / "ner_corpus_profile_per_sentence.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            r = dict(r)
            r["ar_ratio"] = round(r["ar_ratio"], 4)
            r["en_ratio"] = round(r["en_ratio"], 4)
            r["cmi"] = round(r["cmi"], 2)
            w.writerow(r)

    # summary CSV (combo shares serialized as JSON string)
    if summaries:
        sum_cols = [k for k in summaries[0].keys() if k != "entity_combo_share_pct"] + ["entity_combo_share_pct"]
        with open(OUT / "ner_corpus_profile_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=sum_cols)
            w.writeheader()
            for s in summaries:
                row = dict(s)
                row["entity_combo_share_pct"] = json.dumps(row["entity_combo_share_pct"], ensure_ascii=False)
                w.writerow(row)

    (OUT / "ner_corpus_profile_table.tex").write_text(latex_table(summaries), encoding="utf-8")
    print(f"\nwrote {OUT / 'ner_corpus_profile_per_sentence.csv'}")
    print(f"wrote {OUT / 'ner_corpus_profile_summary.csv'}")
    print(f"wrote {OUT / 'ner_corpus_profile_table.tex'}")

    print("\nEntity-type combo shares (% of entity-bearing sentences):")
    for s in summaries:
        print(f"  {s['dataset']}: {s['entity_combo_share_pct']}")


if __name__ == "__main__":
    main()
