"""
refiner_clean_test.py — CLEAN within-sentence test of the refiner (Test 6a, done right).
=========================================================================================
The Step 4 cross-run test was confounded (raw & fixed are different generations).
This test removes that confound: it takes the SAME weak sentences, and for each one
  before = re-score the original sentence  (fresh, this session)
  after  = refine it, then re-score the refined sentence (fresh, this session)
so before vs after measure the refiner's TRUE effect on the same input.

Runs on weak sentences (raw score < BAR), capped by --limit to control API cost.
Every refined attempt is recorded (no guardrail selection bias).

Usage:
  python refiner_clean_test.py                 # default: 30 weakest sentences, bar 7.0
  python refiner_clean_test.py --limit 20 --bar 7.0
Outputs to step4_final_picture/:
  refiner_within_sentence.csv  + a printed summary
"""
import argparse, importlib, json, math, os, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "step1_raw_data" / "Arabic.jsonl"
OUT = HERE / "step4_final_picture"
OUT.mkdir(parents=True, exist_ok=True)

# --- env + SSL (corporate proxy) ---
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


def build_summary(rec, j, score):
    flu = (rec.get("fluency_results_per_instances") or [{}])[j] if j < len(rec.get("fluency_results_per_instances") or []) else {}
    nat = (rec.get("naturalness_results_per_instances") or [{}])[j] if j < len(rec.get("naturalness_results_per_instances") or []) else {}
    csr = (rec.get("cs_ratio_results_per_instances") or [{}])[j] if j < len(rec.get("cs_ratio_results_per_instances") or []) else {}
    soc = (rec.get("social_cultural_results_per_instances") or [{}])[j] if j < len(rec.get("social_cultural_results_per_instances") or []) else {}
    txt = rec.get("data_generation_result", [])[j]
    return (f"Sentence index: {j}\nOriginal sentence: {txt}\nFailure reason: quality_fail\n"
            f"Sentence score: {score}\nFluency: {flu}\nNaturalness: {nat}\n"
            f"CS ratio: {csr}\nSocial cultural: {soc}\nTask validation: {{}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--bar", type=float, default=7.0)
    a = ap.parse_args()

    recs = load(RAW)
    # collect weak sentences (raw score < bar)
    weak = []
    for idx, r in enumerate(recs):
        for j, s in enumerate(r.get("sentence_scores", [])):
            if float(s) < a.bar:
                weak.append((idx, r, j, float(s)))
    weak.sort(key=lambda x: x[3])          # weakest first
    weak = weak[:a.limit]

    refiner = ne.REFINER_PROMPT | ChatOpenAI(
        model=ne.MODEL, temperature=0.1, base_url=ne.API_BASE, api_key=ne.API_KEY
    ).with_structured_output(GenerationResponse)

    print(f"Testing {len(weak)} weak sentences (raw score < {a.bar}). ~9 API calls each.\n")
    rows = []
    for n, (idx, rec, j, saved) in enumerate(weak, 1):
        orig = rec["data_generation_result"][j]
        base = dict(rec)
        try:
            before = ne._rescore_single_sentence(orig, base)             # fresh before
            summ = build_summary(rec, j, round(before, 3))
            resp = refiner.invoke({"summary": summ})
            cand = (resp.get("instances") or [None])[0] if isinstance(resp, dict) else None
            if not (isinstance(cand, str) and cand.strip()):
                print(f"  {n}/{len(weak)} scen{idx} s{j}: refiner returned nothing, skipped")
                continue
            cand = cand.strip()
            after = ne._rescore_single_sentence(cand, base)              # fresh after
            rows.append({"scenario": idx, "sentence_pos": j,
                         "before": round(before, 3), "after": round(after, 3),
                         "delta": round(after - before, 3),
                         "improved": after > before,
                         "before_text": orig, "after_text": cand})
            print(f"  {n}/{len(weak)} scen{idx} s{j}: {round(before,2)} -> {round(after,2)}  (delta {round(after-before,2)})")
        except Exception as e:
            print(f"  {n}/{len(weak)} scen{idx} s{j}: ERROR {type(e).__name__}: {str(e)[:80]}")

    if not rows:
        print("\nNo results.")
        return

    deltas = [r["delta"] for r in rows]
    improved = sum(r["improved"] for r in rows)
    import csv
    with open(OUT / "refiner_within_sentence.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 60)
    print("CLEAN REFINER TEST (same sentence, before vs after)")
    print("=" * 60)
    print(f"  sentences tested:     {len(rows)}")
    print(f"  mean before:          {round(float(np.mean([r['before'] for r in rows])),3)}")
    print(f"  mean after:           {round(float(np.mean([r['after'] for r in rows])),3)}")
    print(f"  mean delta (after-before): {round(float(np.mean(deltas)),3)}")
    print(f"  median delta:         {round(float(np.median(deltas)),3)}")
    print(f"  improved:             {improved}/{len(rows)} ({round(100*improved/len(rows),1)}%)")
    print(f"  sign-test p:          {round(sign_test_p(improved, len(rows)),4)}")
    print(f"\n  -> refiner_within_sentence.csv")
    print("  READ: positive mean delta + most improved + small p = refiner genuinely helps.")


if __name__ == "__main__":
    main()
