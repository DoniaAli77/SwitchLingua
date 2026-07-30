"""
profile_topic_corpus.py — intrinsic evaluation of the generated TOPIC-540 corpus
================================================================================
Topic analogue of profile_generated_corpora.py, adding what the 9-class task needs:

  A. Overall profile of TOPIC-540 (same 18 metrics as the sentiment profile)
  B. PER-CLASS breakdown (9 topics x length / AR% / CMI / switch points / quality)
  C. Side-by-side vs the REAL ArEnTC corpus (domain-compatibility check, the topic
     analogue of the GEN-vs-EESA comparison) incl. vocabulary overlap
  D. Lexical stats: type/token ratio, top English insertions (the CS "anchor" words)

Reuses the exact tokenizer/CMI definitions from profile_generated_corpora.py so the
numbers are directly comparable with the sentiment tables.

Outputs -> multi-agent-bert/data/Topic/generated/merged/corpus_profile/
  topic_profile_overall.csv · topic_profile_per_class.csv
  topic_vs_arentc.csv · topic_top_english_terms.csv · topic_profile_tables.tex
"""
import csv, json, pathlib, random, re, statistics, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "switchlingua"))
from profile_generated_corpora import profile_sentence, p90, norm_text, AR_LETTER, EN_LETTER  # shared defs

TOPIC_GEN = ROOT / "multi-agent-bert" / "data" / "Topic" / "generated" / "merged" / "switchlingua_topic_train_540_60perlabel.jsonl"
ARENTC = ROOT / "multi-agent-bert" / "data" / "Topic" / "processed" / "ARENTCV2" / "train.jsonl"
OUT = TOPIC_GEN.parent / "corpus_profile"
ARENTC_SAMPLE = 5000   # sample for speed; set 0 to use all
CLASSES = ["business", "education", "finance", "health", "medical", "shopping", "social", "sports", "tech"]


def load(path, limit=0, seed=7):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit and len(rows) > limit:
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return rows


def agg(rows):
    """Aggregate profile over a list of {text,label,...} rows."""
    ps = [profile_sentence(str(r.get("text", ""))) for r in rows]
    n = len(ps)
    lens = [p["n_tokens"] for p in ps]
    tot = sum(lens)
    ar = sum(p["ar_tokens"] for p in ps)
    en = sum(p["en_tokens"] for p in ps)
    other = sum(p["other_tokens"] for p in ps)
    cmis = [p["cmi"] for p in ps]
    sw = [p["switch_points"] for p in ps]
    q = [float(r["quality_score"]) for r in rows if str(r.get("quality_score", "")).strip()]
    return {
        "n": n, "tokens": tot,
        "len_mean": round(statistics.mean(lens), 2), "len_median": statistics.median(lens), "len_p90": p90(lens),
        "ar_pct": round(100 * ar / tot, 2), "en_pct": round(100 * en / tot, 2),
        "other_pct": round(100 * other / tot, 2),
        "cmi_mean": round(statistics.mean(cmis), 2), "cmi_median": round(statistics.median(cmis), 2),
        "switch_mean": round(statistics.mean(sw), 2), "switch_median": statistics.median(sw),
        "cs_valid_pct": round(100 * sum(p["is_cs"] for p in ps) / n, 1),
        "dups": len(rows) - len({norm_text(str(r.get("text", ""))) for r in rows}),
        "quality_mean": round(statistics.mean(q), 3) if q else "",
    }


def en_terms(rows):
    """English (Latin) word types, lowercased — the code-switch insertions."""
    c = Counter()
    for r in rows:
        for tok in str(r.get("text", "")).split():
            if EN_LETTER.search(tok) and not AR_LETTER.search(tok):
                w = re.sub(r"[^A-Za-z']", "", tok).lower()
                if len(w) > 1:
                    c[w] += 1
    return c


def ttr(rows):
    toks = [t.lower() for r in rows for t in str(r.get("text", "")).split()]
    return round(len(set(toks)) / len(toks), 4) if toks else 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gen = load(TOPIC_GEN)
    real = load(ARENTC, ARENTC_SAMPLE)
    print(f"TOPIC-540: {len(gen)} rows | ArEnTC(sample): {len(real)} rows\n")

    # ---- A. overall ----
    g_all, r_all = agg(gen), agg(real)
    with open(OUT / "topic_profile_overall.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["metric", "TOPIC-540", "ArEnTC(real)"])
        for k in g_all:
            w.writerow([k, g_all[k], r_all.get(k, "")])
    print("== A. OVERALL ==")
    for k in g_all:
        print(f"  {k:15} gen={str(g_all[k]):>8}   real={str(r_all.get(k,'')):>8}")

    # ---- B. per class ----
    print("\n== B. PER-CLASS (TOPIC-540) ==")
    print(f"  {'class':10} {'n':>4} {'len':>6} {'AR%':>6} {'CMI':>6} {'switch':>7} {'qual':>6}")
    per_class = []
    for lab in CLASSES:
        rows = [r for r in gen if r.get("label") == lab]
        if not rows:
            continue
        a = agg(rows)
        per_class.append({"label": lab, **a})
        print(f"  {lab:10} {a['n']:>4} {a['len_mean']:>6} {a['ar_pct']:>6} {a['cmi_mean']:>6} "
              f"{a['switch_mean']:>7} {str(a['quality_mean']):>6}")
    with open(OUT / "topic_profile_per_class.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class[0].keys())); w.writeheader(); w.writerows(per_class)

    # per-class of the real corpus (for the comparison table)
    real_per_class = {}
    for lab in CLASSES:
        rows = [r for r in real if r.get("label") == lab]
        if rows:
            real_per_class[lab] = agg(rows)

    # ---- C. gen vs real ----
    gen_vocab = {t.lower() for r in gen for t in str(r["text"]).split()}
    real_vocab = {t.lower() for r in real for t in str(r["text"]).split()}
    gen_en, real_en = en_terms(gen), en_terms(real)
    overlap = len(gen_vocab & real_vocab)
    en_overlap = len(set(gen_en) & set(real_en))
    print("\n== C. TOPIC-540 vs ArEnTC ==")
    print(f"  AR%      gen {g_all['ar_pct']:>6}  real {r_all['ar_pct']:>6}   Δ {g_all['ar_pct']-r_all['ar_pct']:+.2f}")
    print(f"  CMI      gen {g_all['cmi_mean']:>6}  real {r_all['cmi_mean']:>6}   Δ {g_all['cmi_mean']-r_all['cmi_mean']:+.2f}")
    print(f"  switch   gen {g_all['switch_mean']:>6}  real {r_all['switch_mean']:>6}   Δ {g_all['switch_mean']-r_all['switch_mean']:+.2f}")
    print(f"  length   gen {g_all['len_mean']:>6}  real {r_all['len_mean']:>6}   Δ {g_all['len_mean']-r_all['len_mean']:+.2f}")
    print(f"  CS-valid gen {g_all['cs_valid_pct']:>6}% real {r_all['cs_valid_pct']:>6}%")
    print(f"  TTR      gen {ttr(gen):>6}  real {ttr(real):>6}")
    print(f"  vocab overlap: {overlap} types ({100*overlap/len(gen_vocab):.1f}% of gen vocab covered by real)")
    print(f"  English-term overlap: {en_overlap} ({100*en_overlap/len(gen_en):.1f}% of gen EN types seen in real)")
    with open(OUT / "topic_vs_arentc.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["metric", "TOPIC-540", "ArEnTC", "delta"])
        for k in ["len_mean", "ar_pct", "en_pct", "cmi_mean", "cmi_median", "switch_mean", "cs_valid_pct"]:
            gv, rv = g_all[k], r_all[k]
            w.writerow([k, gv, rv, round(gv - rv, 2)])
        w.writerow(["ttr", ttr(gen), ttr(real), ""])
        w.writerow(["vocab_types", len(gen_vocab), len(real_vocab), ""])
        w.writerow(["vocab_overlap_types", overlap, "", ""])
        w.writerow(["gen_vocab_covered_by_real_pct", round(100 * overlap / len(gen_vocab), 1), "", ""])
        w.writerow(["en_type_overlap_pct", round(100 * en_overlap / len(gen_en), 1), "", ""])

    # ---- D. top English insertions ----
    print("\n== D. TOP ENGLISH INSERTIONS (gen) ==")
    print("  " + ", ".join(f"{w}({c})" for w, c in gen_en.most_common(15)))
    print("== D. TOP ENGLISH INSERTIONS (real ArEnTC) ==")
    print("  " + ", ".join(f"{w}({c})" for w, c in real_en.most_common(15)))
    with open(OUT / "topic_top_english_terms.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["rank", "gen_term", "gen_count", "real_term", "real_count"])
        gm, rm = gen_en.most_common(30), real_en.most_common(30)
        for i in range(30):
            w.writerow([i + 1,
                        gm[i][0] if i < len(gm) else "", gm[i][1] if i < len(gm) else "",
                        rm[i][0] if i < len(rm) else "", rm[i][1] if i < len(rm) else ""])

    # ---- LaTeX ----
    L = ["% Auto-generated by profile_topic_corpus.py", "\\begin{table}[t]", "\\centering",
         "\\caption{Intrinsic profile of the generated TOPIC-540 corpus and the real ArEnTC corpus.}",
         "\\label{tab:topic-corpus-profile}", "\\begin{tabular}{lrr}", "\\toprule",
         "Metric & TOPIC-540 (generated) & ArEnTC (real) \\\\", "\\midrule",
         f"Examples & {g_all['n']} & {r_all['n']} (sample) \\\\",
         f"Total tokens & {g_all['tokens']:,} & {r_all['tokens']:,} \\\\",
         f"Sentence length (mean / median / P90) & {g_all['len_mean']} / {g_all['len_median']} / {g_all['len_p90']} & "
         f"{r_all['len_mean']} / {r_all['len_median']} / {r_all['len_p90']} \\\\",
         f"Arabic tokens (\\%) & {g_all['ar_pct']} & {r_all['ar_pct']} \\\\",
         f"English tokens (\\%) & {g_all['en_pct']} & {r_all['en_pct']} \\\\",
         f"CMI (mean / median) & {g_all['cmi_mean']} / {g_all['cmi_median']} & {r_all['cmi_mean']} / {r_all['cmi_median']} \\\\",
         f"Switch points (mean / median) & {g_all['switch_mean']} / {g_all['switch_median']} & "
         f"{r_all['switch_mean']} / {r_all['switch_median']} \\\\",
         f"Valid code-switched (\\%) & {g_all['cs_valid_pct']} & {r_all['cs_valid_pct']} \\\\",
         f"Type/token ratio & {ttr(gen)} & {ttr(real)} \\\\",
         f"Duplicates & {g_all['dups']} & {r_all['dups']} \\\\",
         "\\bottomrule", "\\end{tabular}", "\\end{table}", "",
         "\\begin{table}[t]", "\\centering",
         "\\caption{Per-class intrinsic profile of TOPIC-540 (60 sentences per class).}",
         "\\label{tab:topic-per-class}", "\\begin{tabular}{lrrrrr}", "\\toprule",
         "Class & Len (mean) & Arabic \\% & CMI & Switches & Quality \\\\", "\\midrule"]
    for pc in per_class:
        L.append(f"{pc['label']} & {pc['len_mean']} & {pc['ar_pct']} & {pc['cmi_mean']} & "
                 f"{pc['switch_mean']} & {pc['quality_mean']} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (OUT / "topic_profile_tables.tex").write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote 4 CSVs + topic_profile_tables.tex -> {OUT}")


if __name__ == "__main__":
    main()
