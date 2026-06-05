"""
run_csratio_partial_validation.py — PARTIAL CS-ratio MEASUREMENT validation (Test 4, partial).
==============================================================================================
Compares CS-ratio MEASUREMENT methods on a FIXED 30-sentence set. This is NOT a generation
comparison and runs NO generation pipeline — it only measures the same fixed sentences with:
  1. Modified DETERMINISTIC counter (compute_true_cs_stats)        — our method (0 variance, free)
  2. Original-style LLM-ONLY counting (gpt-4o-mini), repeated 3x   — baseline method (instability)
  3. Human/manual token counts                                    — columns blank, PENDING

(A separate script, run_csratio_validation.py, does the full per-sentence A/B/C post-hoc pass;
this partial test is intentionally kept separate and does not touch generation prompts/validator/NER.)

Inputs:  experiments/outputs/switchlingua/csratio/csratio_validation_set.csv
Outputs: csratio_validation_details.jsonl, csratio_llm_repeats.jsonl,
         csratio_partial_summary.csv, csratio_partial_report.md
"""
import csv, importlib, json, os, pathlib, re, statistics, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
CS = ROOT / "experiments" / "outputs" / "switchlingua" / "csratio"
SET = CS / "csratio_validation_set.csv"
LLM_TEMP = 0.7      # reported in the output; the deterministic method has ZERO variance regardless
LLM_REPEATS = 3

# env + SSL (corporate proxy: verify off, matches the rest of the pipeline)
import dotenv
dotenv.load_dotenv(str(ROOT / "Modified_Version" / ".env"), override=True)
import ssl, httpx
ssl._create_default_https_context = ssl._create_unverified_context
_o = httpx.Client.__init__
httpx.Client.__init__ = lambda self, *a, **k: (k.setdefault("verify", False), k.setdefault("timeout", 60.0), _o(self, *a, **k))[-1]
if str(MODIFIED_CORE) not in sys.path:
    sys.path.insert(0, str(MODIFIED_CORE))
for m in ("utils", "node_engine"):
    sys.modules.pop(m, None)
importlib.invalidate_caches()
from utils import compute_true_cs_stats
import node_engine as ne
from langchain_openai import ChatOpenAI

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=LLM_TEMP, base_url=ne.API_BASE, api_key=ne.API_KEY)

LLM_PROMPT = (
    "You are a token-counting tool. Count the word tokens in the sentence below by language.\n"
    "Return STRICT JSON only, no prose:\n"
    '{"arabic_token_count": <int>, "english_token_count": <int>, "other_token_count": <int>, "is_code_switched": <true|false>}\n'
    "- arabic_token_count = Arabic-script word tokens.\n"
    "- english_token_count = English/Latin-script word tokens.\n"
    "- other_token_count = numbers, symbols, or tokens in neither language.\n"
    "- is_code_switched = true if the sentence mixes Arabic AND English, else false.\n\n"
    "Sentence: "
)

AR = r"[؀-ۿݐ-ݿࢠ-ࣿ]"


def det_other_count(text):
    # derived 'other' = whitespace tokens with no Arabic and no Latin letter (numbers/symbols);
    # compute_true_cs_stats natively counts only ar/en, so this is a separate derived figure.
    return sum(1 for tok in text.split() if re.search(r"\S", tok) and not re.search(AR + r"|[A-Za-z]", tok))


def parse_llm(raw):
    try:
        s = raw.content if hasattr(raw, "content") else str(raw)
        a, b = s.find("{"), s.rfind("}")
        o = json.loads(s[a:b + 1]) if a != -1 and b > a else json.loads(s)
        return {"arabic": int(o.get("arabic_token_count", 0)), "english": int(o.get("english_token_count", 0)),
                "other": int(o.get("other_token_count", 0)), "is_cs": bool(o.get("is_code_switched", False))}
    except Exception:
        return None


def ratio(ar, en):
    return round(100 * ar / (ar + en), 2) if (ar + en) else 0.0


def main():
    rows = list(csv.DictReader(SET.open(encoding="utf-8-sig")))
    details, repeats_log = [], []
    have_human = any((r.get("human_arabic_token_count") or "").strip() for r in rows)

    for r in rows:
        text = r["text"]
        st = compute_true_cs_stats(text)
        det = {"arabic": st["cs_ar_count"], "english": st["cs_en_count"], "other": det_other_count(text),
               "ar_ratio": round(st["cs_ar_ratio"], 2), "en_ratio": round(st["cs_en_ratio"], 2),
               "is_cs": bool(st["is_code_switched"])}

        reps = []
        for _ in range(LLM_REPEATS):
            try:
                p = parse_llm(_llm.invoke(LLM_PROMPT + text))
            except Exception:
                p = None
            if p is not None:
                p["ar_ratio"] = ratio(p["arabic"], p["english"])
                reps.append(p)
        repeats_log.append({"sample_id": r["sample_id"], "text": text, "repeats": reps})

        def col(k):
            return [x[k] for x in reps]
        llm_mean = {k: round(statistics.mean(col(k)), 2) for k in ("arabic", "english", "other", "ar_ratio")} if reps else {}
        llm_std = {k: round(statistics.pstdev(col(k)), 3) for k in ("arabic", "english", "ar_ratio")} if len(reps) > 1 else {}
        votes = [x["is_cs"] for x in reps]
        llm_majority_cs = (sum(votes) > len(votes) / 2) if votes else None
        repeats_disagree = (len(set(votes)) > 1) or bool(reps and (max(col("arabic")) - min(col("arabic")) > 1))
        ratio_diff = round(abs(det["ar_ratio"] - llm_mean.get("ar_ratio", det["ar_ratio"])), 2) if reps else None
        cs_mismatch = (llm_majority_cs is not None and det["is_cs"] != llm_majority_cs)

        details.append({
            "sample_id": r["sample_id"], "source": r["source"],
            "is_controlled_edge_case": r["is_controlled_edge_case"], "edge_case_type": r["edge_case_type"],
            "text": text, "deterministic": det, "llm_mean": llm_mean, "llm_std": llm_std,
            "llm_majority_is_cs": llm_majority_cs, "llm_n_ok": len(reps),
            "llm_repeats_disagree": bool(repeats_disagree),
            "det_vs_llm_ar_ratio_abs_diff": ratio_diff, "det_vs_llm_is_cs_mismatch": bool(cs_mismatch),
            "human_pending": not have_human,
        })

    with open(CS / "csratio_validation_details.jsonl", "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(CS / "csratio_llm_repeats.jsonl", "w", encoding="utf-8") as f:
        for d in repeats_log:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ---- aggregate metrics (human PENDING) ----
    n = len(details)
    std_lists = [d for d in details if d["llm_std"]]
    mean_std_ar = round(statistics.mean([d["llm_std"]["arabic"] for d in std_lists]), 3) if std_lists else None
    mean_std_ratio = round(statistics.mean([d["llm_std"]["ar_ratio"] for d in std_lists]), 3) if std_lists else None
    n_repeats_disagree = sum(1 for d in details if d["llm_repeats_disagree"])
    n_cs_mismatch = sum(1 for d in details if d["det_vs_llm_is_cs_mismatch"])
    diffs = [d["det_vs_llm_ar_ratio_abs_diff"] for d in details if d["det_vs_llm_ar_ratio_abs_diff"] is not None]
    mean_ratio_diff = round(statistics.mean(diffs), 2) if diffs else None
    edge = [d for d in details if d["is_controlled_edge_case"] == "yes"]
    mono = [d for d in edge if d["edge_case_type"] in ("fully_arabic", "fully_english")]
    det_mono_ok = sum(1 for d in mono if d["deterministic"]["is_cs"] is False)
    llm_mono_ok = sum(1 for d in mono if d["llm_majority_is_cs"] is False)

    summary = [
        ("n_sentences", n), ("llm_model", "gpt-4o-mini"), ("llm_temperature", LLM_TEMP), ("llm_repeats", LLM_REPEATS),
        ("human_counts", "PENDING (blank)" if not have_human else "present"),
        ("deterministic_variance", "0 (exact, reproducible, free)"),
        ("mean_llm_std_arabic_count", mean_std_ar), ("mean_llm_std_arabic_ratio_pct", mean_std_ratio),
        ("sentences_llm_repeats_disagree", f"{n_repeats_disagree}/{n}"),
        ("det_vs_llm_is_cs_mismatch", f"{n_cs_mismatch}/{n}"),
        ("mean_det_vs_llm_arabic_ratio_abs_diff_pct", mean_ratio_diff),
        ("monolingual_edge_cases", len(mono)),
        ("det_monolingual_correct", f"{det_mono_ok}/{len(mono)}"),
        ("llm_monolingual_correct", f"{llm_mono_ok}/{len(mono)}"),
    ]
    with open(CS / "csratio_partial_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["metric", "value"]); w.writerows(summary)

    # ---- report ----
    disagree_ex = sorted([d for d in details if d["llm_repeats_disagree"]],
                         key=lambda d: -(d["llm_std"].get("ar_ratio", 0) if d["llm_std"] else 0))[:6]
    detllm_ex = [d for d in details if d["det_vs_llm_is_cs_mismatch"] or (d["det_vs_llm_ar_ratio_abs_diff"] or 0) >= 15][:6]
    L = ["# CS-ratio Measurement Validation — PARTIAL (Test 4)\n",
         "**Measurement-method comparison on a FIXED 30-sentence set — NOT an Original-vs-Modified generation "
         "comparison, and no generation pipeline was run.** Methods: (1) our deterministic counter "
         "`compute_true_cs_stats`, (2) original-style LLM-only counting "
         f"(gpt-4o-mini, temperature {LLM_TEMP}, {LLM_REPEATS} repeats), (3) human counts (PENDING).\n",
         "## Status",
         "- **Human-reference accuracy metrics are PENDING** (manual token counts not yet filled).",
         "- This partial run evaluates **LLM instability** and **deterministic-vs-LLM disagreement** only.",
         "- **Final accuracy claims (MAE, detection accuracy, boundary error) require the manual human token counts.**\n",
         "## Results (now)\n", "| metric | value |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in summary]
    L += ["\n## LLM instability — the 3 repeats disagree on these"]
    for d in disagree_ex:
        L.append(f"- `{d['sample_id']}` ({d['edge_case_type'] or d['source']}): ar_count std="
                 f"{d['llm_std'].get('arabic')}, ratio std={d['llm_std'].get('ar_ratio')}%  | det ar%="
                 f"{d['deterministic']['ar_ratio']} vs llm_mean={d['llm_mean'].get('ar_ratio')}  | {d['text'][:55]}")
    if not disagree_ex:
        L.append("- (none — LLM repeats were stable on this set)")
    L += ["\n## Deterministic vs LLM disagreement"]
    for d in detllm_ex:
        L.append(f"- `{d['sample_id']}` det(is_cs={d['deterministic']['is_cs']}, ar%={d['deterministic']['ar_ratio']}) "
                 f"vs llm(is_cs={d['llm_majority_is_cs']}, ar%={d['llm_mean'].get('ar_ratio')})  | {d['text'][:50]}")
    if not detllm_ex:
        L.append("- (none above the 15-pt / is_cs-flip threshold)")
    L += ["\n## Monolingual detection on controlled edge cases",
          f"- fully-Arabic / fully-English cases: deterministic correct **{det_mono_ok}/{len(mono)}**, "
          f"LLM correct **{llm_mono_ok}/{len(mono)}**.\n",
          "## Method notes",
          "- Deterministic counter = `compute_true_cs_stats` (Arabic vs Latin script token counts): **0 variance, "
          "no API call, exact and reproducible**. 'other' is a derived count (numbers/symbols) for the 3-way view.",
          "- LLM ar_ratio = arabic/(arabic+english), matching the deterministic ratio definition.",
          "- When human counts are filled, re-running adds: Arabic/English/other token-count MAE, ratio MAE, "
          "monolingual + code-switch detection accuracy, and boundary error rate vs the human reference.\n"]
    (CS / "csratio_partial_report.md").write_text("\n".join(L), encoding="utf-8")

    print(f"n={n}  human=PENDING  llm=gpt-4o-mini@{LLM_TEMP}")
    print(f"mean LLM std: arabic_count={mean_std_ar}  arabic_ratio%={mean_std_ratio}")
    print(f"LLM repeats disagree: {n_repeats_disagree}/{n} | det-vs-LLM is_cs mismatch: {n_cs_mismatch}/{n} | mean ratio diff={mean_ratio_diff}%")
    print(f"monolingual edge cases: det {det_mono_ok}/{len(mono)}, llm {llm_mono_ok}/{len(mono)}")
    print(f"wrote details.jsonl, llm_repeats.jsonl, partial_summary.csv, partial_report.md -> {CS}")


if __name__ == "__main__":
    main()
