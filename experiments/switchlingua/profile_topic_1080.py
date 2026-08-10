"""
profile_topic_1080.py — intrinsic evaluation of TOPIC-540 and TOPIC-1080
========================================================================
Same methodology as profile_generated_corpora.py (sentiment) and profile_ner_corpora.py
(NER): identical tokenizer, CMI definition and switch-point definition, imported rather
than re-implemented, so every number is directly comparable across the three tracks.

  A. Overall profile of TOPIC-540, TOPIC-1080 and the NEW-540 half added to build 1080,
     side by side with the real reference corpora (ArEnTC and Silver-1044)
  B. Per-class breakdown for both generated corpora
  C. Vocabulary overlap of each generated corpus against each real corpus
  D. Top English insertions (the code-switch "anchor" words)

CMI here is the same per-sentence definition used throughout: 100 - max(ar%, en%), so 50
is a perfectly balanced sentence and 0 is monolingual.

TTR is reported twice: raw (NOT comparable across corpora of different sizes, since type/
token ratio falls mechanically with length) and ttr_7k, computed on a fixed 7,000-token
random subsample so the corpora can actually be compared.

Outputs -> multi-agent-bert/data/Topic/generated/merged/corpus_profile_1080/
  topic1080_profile_overall.csv · topic1080_profile_per_class.csv
  topic1080_vs_real.csv · topic1080_top_english_terms.csv · topic1080_profile_tables.tex
"""
import csv, json, pathlib, random, statistics, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
SW = ROOT / "experiments" / "switchlingua"
sys.path.insert(0, str(SW))
from profile_topic_corpus import agg, en_terms, load, ARENTC, ARENTC_SAMPLE   # shared defs
from profile_generated_corpora import norm_text                                # shared defs

MAB = ROOT / "multi-agent-bert"
GEN = MAB / "data" / "Topic" / "generated"
T540 = GEN / "merged" / "switchlingua_topic_train_540_60perlabel.jsonl"
T1080 = GEN / "merged" / "switchlingua_topic_train_1080_120perlabel.jsonl"
BATCHES = GEN / "expansion_1080" / "batches"
SILVER = (MAB / "experiments/outputs/multi_agent_bert/experiment_silver_topic540"
          / "silver_primary_1044.jsonl")
OUT = GEN / "merged" / "corpus_profile_1080"
CLASSES = ["business", "education", "finance", "health", "medical",
           "shopping", "social", "sports", "tech"]


def rd(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def ttr(rows, n=None, seed=7):
    toks = [t.lower() for r in rows for t in str(r.get("text", "")).split()]
    if n:
        if len(toks) < n:
            return None
        random.Random(seed).shuffle(toks)
        toks = toks[:n]
    return round(len(set(toks)) / len(toks), 4) if toks else 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t540, t1080 = rd(T540), rd(T1080)
    new540 = [r for b in (1, 2, 3) for r in rd(BATCHES / f"batch{b}_accepted.jsonl")]
    arentc = load(ARENTC, ARENTC_SAMPLE)
    silver = rd(SILVER)

    sets = {"TOPIC-540": t540, "TOPIC-1080": t1080, "new-540": new540,
            "ArEnTC(5k)": arentc, "Silver-1044": silver}
    prof = {k: agg(v) for k, v in sets.items()}
    for k, v in sets.items():
        prof[k]["ttr_raw"] = ttr(v)
        prof[k]["ttr_7k"] = ttr(v, 7000)

    keys = ["n", "tokens", "len_mean", "len_median", "len_p90", "ar_pct", "en_pct",
            "other_pct", "cmi_mean", "cmi_median", "switch_mean", "switch_median",
            "cs_valid_pct", "dups", "quality_mean", "ttr_raw", "ttr_7k"]
    names = list(sets)

    print("== A. OVERALL ==")
    print(f"{'metric':<14}" + "".join(f"{n:>14}" for n in names))
    for k in keys:
        print(f"{k:<14}" + "".join(f"{str(prof[n].get(k, '')):>14}" for n in names))
    with open(OUT / "topic1080_profile_overall.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["metric"] + names)
        for k in keys:
            w.writerow([k] + [prof[n].get(k, "") for n in names])

    # ---- B. per class ----
    print("\n== B. PER CLASS ==")
    print(f"  {'class':<10} {'n':>4} {'len':>6} {'AR%':>6} {'CMI':>6} {'sw':>5} {'qual':>6}"
          f"   |  {'n':>4} {'len':>6} {'AR%':>6} {'CMI':>6} {'sw':>5} {'qual':>6}")
    print(f"  {'':<10} {'---- TOPIC-540 ----':^36}   |  {'---- TOPIC-1080 ----':^36}")
    rows_pc = []
    for lab in CLASSES:
        a = agg([r for r in t540 if r.get("label") == lab])
        b = agg([r for r in t1080 if r.get("label") == lab])
        rows_pc.append({"label": lab,
                        **{f"t540_{k}": a[k] for k in ("n", "len_mean", "ar_pct", "cmi_mean", "switch_mean", "quality_mean")},
                        **{f"t1080_{k}": b[k] for k in ("n", "len_mean", "ar_pct", "cmi_mean", "switch_mean", "quality_mean")}})
        print(f"  {lab:<10} {a['n']:>4} {a['len_mean']:>6} {a['ar_pct']:>6} {a['cmi_mean']:>6} "
              f"{a['switch_mean']:>5} {str(a['quality_mean']):>6}   |  "
              f"{b['n']:>4} {b['len_mean']:>6} {b['ar_pct']:>6} {b['cmi_mean']:>6} "
              f"{b['switch_mean']:>5} {str(b['quality_mean']):>6}")
    with open(OUT / "topic1080_profile_per_class.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_pc[0])); w.writeheader(); w.writerows(rows_pc)

    # ---- C. gen vs real ----
    print("\n== C. GENERATED vs REAL ==")
    vocab = {k: {t.lower() for r in v for t in str(r["text"]).split()} for k, v in sets.items()}
    ent = {k: en_terms(v) for k, v in sets.items()}
    rows_c = []
    for g in ("TOPIC-540", "TOPIC-1080"):
        for real in ("ArEnTC(5k)", "Silver-1044"):
            ov = len(vocab[g] & vocab[real])
            eov = len(set(ent[g]) & set(ent[real]))
            rec = {"generated": g, "real": real,
                   "d_ar_pct": round(prof[g]["ar_pct"] - prof[real]["ar_pct"], 2),
                   "d_cmi": round(prof[g]["cmi_mean"] - prof[real]["cmi_mean"], 2),
                   "d_switch": round(prof[g]["switch_mean"] - prof[real]["switch_mean"], 2),
                   "d_len": round(prof[g]["len_mean"] - prof[real]["len_mean"], 2),
                   "gen_vocab_covered_pct": round(100 * ov / len(vocab[g]), 1),
                   "en_type_overlap_pct": round(100 * eov / len(ent[g]), 1)}
            rows_c.append(rec)
            print(f"  {g:<11} vs {real:<12} dAR {rec['d_ar_pct']:>+7} dCMI {rec['d_cmi']:>+7} "
                  f"dSwitch {rec['d_switch']:>+6} dLen {rec['d_len']:>+6} "
                  f"vocab_cov {rec['gen_vocab_covered_pct']:>5}% EN_overlap {rec['en_type_overlap_pct']:>5}%")
    with open(OUT / "topic1080_vs_real.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_c[0])); w.writeheader(); w.writerows(rows_c)

    # ---- D. top English insertions ----
    print("\n== D. TOP ENGLISH INSERTIONS ==")
    for k in names:
        print(f"  {k:<12} " + ", ".join(f"{w}({c})" for w, c in ent[k].most_common(12)))
    with open(OUT / "topic1080_top_english_terms.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["rank"] + [f"{n}_term/{n}_count" for n in names])
        tops = {n: ent[n].most_common(30) for n in names}
        for i in range(30):
            row = [i + 1]
            for n in names:
                row.append(f"{tops[n][i][0]}({tops[n][i][1]})" if i < len(tops[n]) else "")
            w.writerow(row)

    # ---- LaTeX ----
    L = ["% Auto-generated by profile_topic_1080.py", "\\begin{table}[t]", "\\centering",
         "\\caption{Intrinsic profile of the generated TOPIC-540 and TOPIC-1080 corpora "
         "against the real ArEnTC and Silver corpora.}",
         "\\label{tab:topic-1080-profile}", "\\begin{tabular}{lrrrr}", "\\toprule",
         "Metric & TOPIC-540 & TOPIC-1080 & ArEnTC & Silver-1044 \\\\", "\\midrule"]
    show = [("Examples", "n"), ("Total tokens", "tokens"), ("Sentence length (mean)", "len_mean"),
            ("Arabic tokens (\\%)", "ar_pct"), ("English tokens (\\%)", "en_pct"),
            ("CMI (mean)", "cmi_mean"), ("CMI (median)", "cmi_median"),
            ("Switch points (mean)", "switch_mean"), ("Valid code-switched (\\%)", "cs_valid_pct"),
            ("Type/token ratio (7k sample)", "ttr_7k"), ("Duplicates", "dups")]
    for lbl, k in show:
        L.append(f"{lbl} & {prof['TOPIC-540'][k]} & {prof['TOPIC-1080'][k]} & "
                 f"{prof['ArEnTC(5k)'].get(k, '-')} & {prof['Silver-1044'].get(k, '-')} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (OUT / "topic1080_profile_tables.tex").write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote 4 CSVs + topic1080_profile_tables.tex -> {OUT}")


if __name__ == "__main__":
    main()
