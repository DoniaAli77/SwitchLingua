"""
step6b_refiner_headtohead.py — YOUR refiner vs the ORIGINAL refiner (Test 6b).
==============================================================================
Proves the stronger claim: your refiner improves weak sentences MORE than the
original's. We isolate the one variable that defines your contribution —
FEEDBACK GRANULARITY — holding the prompt, model, and sentence constant.

For each weak sentence:
  before     = re-score the original sentence (fresh)
  YOUR fix   = refine it using THAT sentence's per-sentence scores  -> re-score
  ORIG fix   = refine it using the scenario's AGGREGATE scores      -> re-score
Same sentence, same REFINER_PROMPT, same model — only the feedback differs.
Whichever lifts the sentence more, wins.

Usage:
  python step6b_refiner_headtohead.py --limit 30
Outputs to step4_final_picture/:
  refiner_headtohead.csv  + a printed summary
"""
import argparse, importlib, json, math, os, pathlib, sys, csv
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "step1_raw_data" / "Arabic.jsonl"
OUT = HERE / "step4_final_picture"
OUT.mkdir(parents=True, exist_ok=True)

import dotenv
dotenv.load_dotenv(str(ROOT / "Modified_Version" / ".env"), override=True)
os.environ["API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
os.environ["API_BASE"] = os.environ.get("OPENAI_BASE_URL", "")
import ssl, httpx
ssl._create_default_https_context = ssl._create_unverified_context
_o = httpx.Client.__init__
def _c(self,*a,**k): k.setdefault("verify",False); k.setdefault("timeout",60.0); _o(self,*a,**k)
httpx.Client.__init__ = _c
_ao = httpx.AsyncClient.__init__
def _ac(self,*a,**k): k.setdefault("verify",False); k.setdefault("timeout",60.0); _ao(self,*a,**k)
httpx.AsyncClient.__init__ = _ac

if str(MODIFIED_CORE) not in sys.path:
    sys.path.insert(0, str(MODIFIED_CORE))
for m in ("utils","node_engine","node_models","prompt","mcp_tools","agents","run_french"):
    sys.modules.pop(m, None)
importlib.invalidate_caches()

import node_engine as ne
from node_models import GenerationResponse
from langchain_openai import ChatOpenAI
ne.MODEL = "gpt-4o-mini"


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def sign_test_p(wins, n):
    if n == 0:
        return float("nan")
    return 2 * (1 - _norm_cdf(abs((wins - n/2.0) / math.sqrt(n/4.0))))


def _item(rec, key, j):
    arr = rec.get(key) or []
    return arr[j] if j < len(arr) and isinstance(arr[j], dict) else {}


def summary_yours(rec, j, before):
    """Per-sentence feedback (YOUR refiner): this sentence's own scores."""
    txt = rec["data_generation_result"][j]
    return (f"Sentence index: {j}\nOriginal sentence: {txt}\nFailure reason: quality_fail\n"
            f"Sentence score: {round(before,3)}\n"
            f"Fluency: {_item(rec,'fluency_results_per_instances',j)}\n"
            f"Naturalness: {_item(rec,'naturalness_results_per_instances',j)}\n"
            f"CS ratio: {_item(rec,'cs_ratio_results_per_instances',j)}\n"
            f"Social cultural: {_item(rec,'social_cultural_results_per_instances',j)}\nTask validation: {{}}")


def summary_orig(rec, j):
    """Aggregate feedback (ORIGINAL refiner): the scenario-average scores only."""
    txt = rec["data_generation_result"][j]
    csr = rec.get("cs_ratio_results_per_instances") or []
    cs_vals = [c.get("ratio_score") for c in csr if isinstance(c, dict) and c.get("ratio_score") is not None]
    cs_avg = round(sum(cs_vals)/len(cs_vals), 2) if cs_vals else "n/a"
    return (f"Original sentence: {txt}\n(Scenario-level feedback — averaged over all sentences)\n"
            f"Fluency: {rec.get('fluency_result', {})}\n"
            f"Naturalness: {rec.get('naturalness_result', {})}\n"
            f"CS ratio (avg): {cs_avg}\n"
            f"Social cultural: {rec.get('social_cultural_result', {})}\n"
            f"Scenario score: {rec.get('score')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--bar", type=float, default=7.0)
    a = ap.parse_args()

    recs = load(RAW)
    weak = []
    for idx, r in enumerate(recs):
        for j, s in enumerate(r.get("sentence_scores", [])):
            if float(s) < a.bar:
                weak.append((idx, r, j, float(s)))
    weak.sort(key=lambda x: x[3])
    weak = weak[:a.limit]

    refiner = ne.REFINER_PROMPT | ChatOpenAI(
        model=ne.MODEL, temperature=0.1, base_url=ne.API_BASE, api_key=ne.API_KEY
    ).with_structured_output(GenerationResponse)

    def refine(summary):
        resp = refiner.invoke({"summary": summary})
        cand = (resp.get("instances") or [None])[0] if isinstance(resp, dict) else None
        return cand.strip() if isinstance(cand, str) and cand.strip() else None

    print(f"Head-to-head on {len(weak)} weak sentences (~14 API calls each).\n")
    rows = []
    for n, (idx, rec, j, saved) in enumerate(weak, 1):
        orig = rec["data_generation_result"][j]
        base = dict(rec)
        try:
            before = ne._rescore_single_sentence(orig, base)
            cy = refine(summary_yours(rec, j, before))
            co = refine(summary_orig(rec, j))
            if not cy or not co:
                print(f"  {n}/{len(weak)} scen{idx} s{j}: a refiner returned nothing, skipped")
                continue
            ay = ne._rescore_single_sentence(cy, base)
            ao = ne._rescore_single_sentence(co, base)
            rows.append({"scenario": idx, "sentence_pos": j, "before": round(before,3),
                         "after_YOURS": round(ay,3), "after_ORIG": round(ao,3),
                         "delta_YOURS": round(ay-before,3), "delta_ORIG": round(ao-before,3),
                         "yours_minus_orig": round(ay-ao,3),
                         "winner": "YOURS" if ay > ao else ("ORIG" if ao > ay else "tie")})
            print(f"  {n}/{len(weak)} scen{idx} s{j}: before {round(before,2)} | yours {round(ay,2)} | orig {round(ao,2)} -> {rows[-1]['winner']}")
        except Exception as e:
            print(f"  {n}/{len(weak)} scen{idx} s{j}: ERROR {type(e).__name__}: {str(e)[:80]}")

    if not rows:
        print("\nNo results.")
        return

    with open(OUT / "refiner_headtohead.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    dy = [r["delta_YOURS"] for r in rows]
    do = [r["delta_ORIG"] for r in rows]
    yours_wins = sum(r["winner"] == "YOURS" for r in rows)
    orig_wins = sum(r["winner"] == "ORIG" for r in rows)
    decided = yours_wins + orig_wins
    print("\n" + "=" * 60)
    print("TEST 6b — YOUR refiner vs ORIGINAL refiner")
    print("=" * 60)
    print(f"  sentences:            {len(rows)}")
    print(f"  mean improvement YOURS: +{round(float(np.mean(dy)),3)}")
    print(f"  mean improvement ORIG:  +{round(float(np.mean(do)),3)}")
    print(f"  mean (YOURS - ORIG):    {round(float(np.mean([r['yours_minus_orig'] for r in rows])),3)}")
    print(f"  wins: YOURS {yours_wins} | ORIG {orig_wins} | ties {len(rows)-decided}")
    print(f"  sign-test p (yours vs orig, decided only): {round(sign_test_p(yours_wins, decided),4) if decided else 'n/a'}")
    print(f"\n  -> refiner_headtohead.csv")
    print("  READ: if YOURS mean improvement > ORIG and YOURS wins most, your per-sentence")
    print("        feedback beats the original's aggregate feedback -> the design matters.")


if __name__ == "__main__":
    main()
