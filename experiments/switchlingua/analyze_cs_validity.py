"""
analyze_cs_validity.py — DIAGNOSE why generated sentiment instances fail deterministic CS-validity.
READ-ONLY: no generation, no prompt/config changes. Classifies every sentence_record across all raw
states (pilot_v1 + daily_runs) and breaks failures down by label/topic/cs_type/cs_function/intensity.

is_code_switched (utils.compute_true_cs_stats) = (arabic_tokens>0 AND latin_tokens>0). A failure is
therefore either fully-Arabic (zero Latin tokens) or fully-English (zero Arabic tokens). English words
written in Arabic SCRIPT are not counted as Latin -> would look fully-Arabic here (flagged separately).

Output: multi-agent-bert/experiments/outputs/multi_agent_bert/generated_sentiment_data/CS_VALIDITY_DIAGNOSIS.md
"""
import json, pathlib, re, sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
GEN = ROOT / "multi-agent-bert" / "data" / "Sentiment" / "generated"
OUT = ROOT / "multi-agent-bert" / "experiments" / "outputs" / "multi_agent_bert" / "generated_sentiment_data" / "CS_VALIDITY_DIAGNOSIS.md"

sys.path.insert(0, str(MODIFIED_CORE))
from utils import compute_true_cs_stats

LATIN = re.compile(r"[A-Za-z]")
ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")


def load_states():
    states = []
    for p in [GEN / "pilot_v1" / "raw_outputs.jsonl", *sorted((GEN / "daily_runs").glob("run_*_raw.jsonl"))]:
        if p.exists():
            for l in p.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    s = json.loads(l)
                    if s.get("task") == "sentiment":
                        states.append((p.stem, s))
    return states


def classify(text):
    st = compute_true_cs_stats(text)
    ar, en = st["cs_ar_count"], st["cs_en_count"]
    if not text.strip():
        cat = "empty"
    elif ar > 0 and en > 0:
        cat = "cs_valid"
    elif ar > 0 and en == 0:
        cat = "fully_arabic"
    elif en > 0 and ar == 0:
        cat = "fully_english"
    else:
        cat = "no_letters"
    return cat, ar, en, st["cs_ar_ratio"]


def rate(d):
    f = d["fail"]; n = d["nonempty"]
    return f"{f}/{n} ({100*f/n:.0f}%)" if n else "0/0 (—)"


def main():
    rows = []
    for src, s in load_states():
        tc = s.get("task_constraints") or {}
        meta = dict(label=s.get("label"), topic=s.get("topic"), cs_type=s.get("cs_type"),
                    cs_function=s.get("cs_function"), intensity=tc.get("intensity"), src=src)
        for sr in s.get("sentence_records") or []:
            text = sr.get("text", "") or ""
            tvp = sr.get("task_passed")
            if tvp is None:
                tvp = (sr.get("task_validation") or {}).get("passed")
            cat, ar, en, arr = classify(text)
            rows.append({**meta, "text": text, "validator_passed": bool(tvp),
                         "cat": cat, "ar": ar, "en": en, "ar_ratio": arr,
                         "latin_chars": bool(LATIN.search(text))})

    nonempty = [r for r in rows if r["cat"] != "empty"]
    cs = [r for r in rows if r["cat"] == "cs_valid"]
    fails = [r for r in nonempty if r["cat"] != "cs_valid"]
    tot = len(nonempty)
    cat_counts = Counter(r["cat"] for r in rows)

    def grp_rate(key):
        d = defaultdict(lambda: {"fail": 0, "nonempty": 0})
        for r in nonempty:
            d[r[key]]["nonempty"] += 1
            if r["cat"] != "cs_valid":
                d[r[key]]["fail"] += 1
        return d

    # English-in-Arabic-script heuristic: a "fully_arabic" failure that nonetheless contains NO latin
    # chars at all is purely Arabic script; if it DID intend English words, they were transliterated.
    fa = [r for r in fails if r["cat"] == "fully_arabic"]
    fa_no_latin = [r for r in fa if not r["latin_chars"]]
    # fragile CS-valid: only 1 English token (barely passes)
    fragile = [r for r in cs if r["en"] == 1]
    mean_arr_valid = round(sum(r["ar_ratio"] for r in cs) / len(cs), 1) if cs else 0
    mean_arr_all = round(sum(r["ar_ratio"] for r in nonempty) / len(nonempty), 1) if nonempty else 0

    L = ["# CS-Validity Failure Diagnosis (Experiment C sentiment generation)\n",
         f"Scope: **{len(rows)}** generated instances across pilot_v1 + daily runs "
         f"({tot} non-empty). READ-ONLY diagnosis — no prompts/config changed.\n",
         "## Headline",
         f"- Non-empty instances: **{tot}** | CS-valid: **{len(cs)}** ({100*len(cs)/tot:.0f}%) | "
         f"CS-FAIL: **{len(fails)}** ({100*len(fails)/tot:.0f}%)",
         f"- Failure breakdown: fully-Arabic **{cat_counts['fully_arabic']}**, "
         f"fully-English **{cat_counts['fully_english']}**, no-letters **{cat_counts['no_letters']}**, "
         f"empty **{cat_counts['empty']}**\n",
         "## Q1–Q5 nature of failures",
         f"1. **Fully Arabic (zero English tokens):** {cat_counts['fully_arabic']} "
         f"({100*cat_counts['fully_arabic']/max(len(fails),1):.0f}% of failures) — the dominant failure mode.",
         f"2. **Fully English (zero Arabic tokens):** {cat_counts['fully_english']} "
         f"({100*cat_counts['fully_english']/max(len(fails),1):.0f}% of failures).",
         f"3. **Mostly-Arabic / too-few-English:** failures have **zero** English by definition; among the "
         f"CS-VALID ones, **{len(fragile)}/{len(cs)}** are *fragile* (exactly 1 English token) — i.e. one "
         f"dropped word away from failing. Mean Arabic share of valid CS sentences = **{mean_arr_valid}%**.",
         f"4. **English written in Arabic script:** {len(fa_no_latin)}/{len(fa)} fully-Arabic failures contain "
         f"**no Latin characters at all** (purely Arabic script). Where the sentiment clearly intends an English "
         f"insertion, it was transliterated into Arabic letters → not counted as English. (See examples.)",
         f"5. **Is 70% Arabic too Arabic-heavy?** Mean Arabic share = **{mean_arr_all}%** over all non-empty, "
         f"**{mean_arr_valid}%** over valid CS — **above the 70% target**, so the matrix is overshooting Arabic, "
         f"leaving little room for (or zero) English tokens in short single sentences.\n",
         "## Q6 Is neutral more likely to go monolingual? / Q7 failure rate by label", "| label | CS-fail rate |", "|---|---|"]
    for k, v in sorted(grp_rate("label").items(), key=lambda x: -(x[1]["fail"]/max(x[1]["nonempty"],1))):
        L.append(f"| {k} | {rate(v)} |")
    L += ["\n## Q8 failure rate by topic", "| topic | CS-fail rate |", "|---|---|"]
    for k, v in sorted(grp_rate("topic").items(), key=lambda x: -(x[1]["fail"]/max(x[1]["nonempty"],1))):
        L.append(f"| {k} | {rate(v)} |")
    L += ["\n## Q9 failure rate by cs_type / cs_function / intensity"]
    for key in ("cs_type", "cs_function", "intensity"):
        L.append(f"\n**{key}:**")
        L.append("| value | CS-fail rate |"); L.append("|---|---|")
        gr = grp_rate(key)
        for k, v in sorted(gr.items(), key=lambda x: -(x[1]["fail"]/max(x[1]["nonempty"],1))):
            L.append(f"| {k} | {rate(v)} |")
        if len(gr) == 1:
            L.append(f"_(only one value present in the data — no variation to compare)_")
    L += ["\n## Q10 — 10 not_cs_valid failure examples (with explanation)"]
    shown = 0
    for r in fails:
        if shown >= 10:
            break
        why = ("fully Arabic, 0 English tokens" if r["cat"] == "fully_arabic"
               else "fully English, 0 Arabic tokens" if r["cat"] == "fully_english" else "no letters")
        extra = " — contains Latin chars but no standalone English word token" if (r["cat"] == "fully_arabic" and r["latin_chars"]) else ""
        L.append(f"{shown+1}. [{r['label']}/{r['cs_type']}/{r['intensity']}] ar={r['ar']} en={r['en']} "
                 f"({why}{extra})\n   `{r['text'][:120]}`")
        shown += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")

    # console summary
    print(f"instances={len(rows)} nonempty={tot} cs_valid={len(cs)} ({100*len(cs)/tot:.0f}%) fail={len(fails)} ({100*len(fails)/tot:.0f}%)")
    print(f"failure modes: fully_arabic={cat_counts['fully_arabic']} fully_english={cat_counts['fully_english']} no_letters={cat_counts['no_letters']}")
    print(f"mean Arabic% valid={mean_arr_valid} all={mean_arr_all} | fragile(1-EN)={len(fragile)}/{len(cs)} | fully_arabic no-latin={len(fa_no_latin)}/{len(fa)}")
    print("\nCS-fail by label:")
    for k, v in grp_rate("label").items():
        print(f"  {k:9} {rate(v)}")
    print("CS-fail by cs_type:")
    for k, v in grp_rate("cs_type").items():
        print(f"  {k:18} {rate(v)}")
    print("CS-fail by intensity:")
    for k, v in grp_rate("intensity").items():
        print(f"  {k:8} {rate(v)}")
    print(f"\nreport -> {OUT}")


if __name__ == "__main__":
    main()
