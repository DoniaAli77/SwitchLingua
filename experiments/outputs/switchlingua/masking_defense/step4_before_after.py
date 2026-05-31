"""
step4_before_after.py — The "payoff" picture: does fixing help?
================================================================
Compares the dataset BEFORE fixing (raw, refiner OFF) vs AFTER fixing
(fixed, refiner ON). They are two independent runs of the same scenario
configs, so this is an UNPAIRED comparison of "refiner off" vs "refiner on".

Key question: does turning the refiner on raise the quality floor
(the worst sentence per scenario) and reduce sub-threshold sentences?

Outputs to step4_final_picture/:
  before_after_summary.csv          — headline metrics (raw vs fixed)
  worst_sentence_per_scenario.csv   — min score per scenario (for an Excel chart)
  + an ASCII comparison printed to screen
Reports the REAL result, positive or not.
"""
import csv
import math
import pathlib
import numpy as np
import json

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "step1_raw_data" / "Arabic.jsonl"
FIXED = HERE / "step1_fixed_data" / "Arabic.jsonl"
OUT = HERE / "step4_final_picture"
OUT.mkdir(parents=True, exist_ok=True)
BAR = 7.0


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def mann_whitney_p(a, b):
    a, b = list(a), list(b)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")
    allv = np.array(a + b, float)
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv)); sv = allv[order]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    _, counts = np.unique(allv, return_counts=True)
    n = n1 + n2
    tie = (counts ** 3 - counts).sum()
    sigma = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))) if n > 1 else 0
    if sigma == 0:
        return float("nan")
    return 2 * (1 - _norm_cdf(abs((u1 - n1 * n2 / 2.0) / sigma)))


def collect(recs):
    all_scores, mins = [], []
    fully_accepted = 0
    n_scen = 0
    refined = 0  # sentences that got >=1 refine attempt (fixed run only)
    for r in recs:
        s = r.get("sentence_scores", [])
        if not isinstance(s, list) or not s:
            continue
        n_scen += 1
        all_scores += [float(x) for x in s]
        mins.append(min(float(x) for x in s))
        if all(float(x) >= BAR for x in s):
            fully_accepted += 1
        for rc in r.get("instance_refine_counts", []) or []:
            if rc and int(rc) >= 1:
                refined += 1
    return {
        "scenarios": n_scen,
        "sentences": len(all_scores),
        "all_scores": all_scores,
        "mins": mins,
        "mean_sentence": float(np.mean(all_scores)) if all_scores else 0,
        "mean_worst_per_scenario": float(np.mean(mins)) if mins else 0,
        "pct_below_bar": 100 * sum(x < BAR for x in all_scores) / len(all_scores) if all_scores else 0,
        "pct_scenarios_fully_accepted": 100 * fully_accepted / n_scen if n_scen else 0,
        "sentences_refined": refined,
    }


def ascii_hist(label, scores):
    bins = [(6.0, 6.5), (6.5, 7.0), (7.0, 7.5), (7.5, 8.0), (8.0, 10.01)]
    print(f"  {label}:")
    for lo, hi in bins:
        c = sum(lo <= x < hi for x in scores)
        bar = "#" * c
        print(f"    {lo:>4.1f}-{hi:>4.1f} | {bar} {c}")


def main():
    raw = collect(load(RAW))
    fixed = collect(load(FIXED))
    p = mann_whitney_p(raw["all_scores"], fixed["all_scores"])

    rows = []
    def add(metric, r, fx, better):
        rows.append({"metric": metric, "before_raw": round(r, 3) if isinstance(r, float) else r,
                     "after_fixed": round(fx, 3) if isinstance(fx, float) else fx,
                     "change": round(fx - r, 3) if isinstance(r, (int, float)) else "",
                     "better_direction": better})
    add("mean sentence score", raw["mean_sentence"], fixed["mean_sentence"], "higher")
    add("mean WORST sentence per scenario", raw["mean_worst_per_scenario"], fixed["mean_worst_per_scenario"], "higher")
    add(f"% sentences below bar {BAR}", raw["pct_below_bar"], fixed["pct_below_bar"], "lower")
    add(f"% scenarios fully accepted (all>={BAR})", raw["pct_scenarios_fully_accepted"], fixed["pct_scenarios_fully_accepted"], "higher")

    with open(OUT / "before_after_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "before_raw", "after_fixed", "change", "better_direction"])
        w.writeheader(); w.writerows(rows)

    # worst-sentence-per-scenario lists (for an Excel chart) — padded to same length
    with open(OUT / "worst_sentence_per_scenario.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["raw_worst", "fixed_worst"])
        for i in range(max(len(raw["mins"]), len(fixed["mins"]))):
            w.writerow([round(raw["mins"][i], 3) if i < len(raw["mins"]) else "",
                        round(fixed["mins"][i], 3) if i < len(fixed["mins"]) else ""])

    print("=" * 64)
    print("STEP 4 — BEFORE (refiner off) vs AFTER (refiner on)")
    print("=" * 64)
    print(f"  scenarios: raw {raw['scenarios']} | fixed {fixed['scenarios']}")
    print(f"  sentences: raw {raw['sentences']} | fixed {fixed['sentences']}")
    print(f"  sentences refined in fixed run: {fixed['sentences_refined']}\n")
    print(f"  {'metric':42s} {'before':>8} {'after':>8} {'change':>8}")
    print("  " + "-" * 70)
    for r in rows:
        print(f"  {r['metric']:42s} {r['before_raw']:>8} {r['after_fixed']:>8} {str(r['change']):>8}  (want {r['better_direction']})")
    print(f"\n  Mann-Whitney p (sentence scores, raw vs fixed): {round(p,4) if p==p else None}\n")

    print("  Worst-sentence-per-scenario distribution:")
    ascii_hist("BEFORE (raw)", raw["mins"])
    ascii_hist("AFTER (fixed)", fixed["mins"])

    print("\n  Files: before_after_summary.csv, worst_sentence_per_scenario.csv")
    print("  Read: if 'after' raises the worst-sentence floor and cuts % below bar,")
    print("        fixing helped. If barely changed, the refiner adds little on this model (honest result).")


if __name__ == "__main__":
    main()
