"""
test_per_sentence_vs_scenario.py
=================================
Thesis Experiment — Prove that per-sentence scoring is more accurate than
per-scenario (aggregate) scoring for quality-gated refinement.

Four tests:
  Test 1 — Policy Disagreement across thresholds
           For each scenario, compare:
             Policy-A (original): refine if agg < threshold
             Policy-B (modified): refine if ANY sentence < threshold
           Show cases where they disagree at each threshold.

  Test 2 — Intra-scenario score variance
           Compute spread (max-min), std-dev, and CV per scenario.
           High spread = aggregate is misleading.

  Test 3 — Masking simulation
           Inject synthetic "masking" records: one bad sentence hidden by
           good neighbours. Show that Policy-A misses them, Policy-B catches them.

  Test 4 — Refiner targeting efficiency
           For each refine event in actual data, show how many sentences
           the original refiner would re-generate vs the modified refiner.
           Fewer = more efficient.

Outputs (CSV):
  experiments/outputs/switchlingua/per_sentence/
    policy_disagreement.csv
    score_variance.csv
    masking_simulation.csv
    refiner_efficiency.csv
    summary_table.csv

Usage:
    python experiments/switchlingua/test_per_sentence_vs_scenario.py
"""

import csv
import pathlib
import statistics
import sys
import json

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments" / "outputs" / "switchlingua" / "per_sentence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSONL_B = ROOT / "experiments" / "outputs" / "switchlingua" / "system_b_modified_mini" / "Arabic.jsonl"

THRESHOLDS = [7.0, 7.3, 7.5, 7.8, 8.0]


def load_jsonl(path):
    import jsonlines
    with jsonlines.open(path) as f:
        return list(f)


# ──────────────────────────────────────────────────────────────────────────────
# TEST 1 — Policy disagreement
# ──────────────────────────────────────────────────────────────────────────────
def test_policy_disagreement(records, thresholds):
    rows = []
    for thr in thresholds:
        agree = disagree_a_refines_more = disagree_b_refines_more = 0
        cases = []
        for i, r in enumerate(records):
            scores = r.get("sentence_scores", [])
            agg = r.get("score", 0) or 0
            if not scores:
                continue
            policy_a = agg < thr                  # original: whole scenario
            policy_b = any(s < thr for s in scores)  # modified: any sentence

            if policy_a == policy_b:
                agree += 1
            elif policy_a and not policy_b:
                # Original refines unnecessarily (all sentences are fine)
                disagree_a_refines_more += 1
                cases.append({
                    "record": i, "threshold": thr,
                    "agg": round(agg, 3),
                    "min_sentence": round(min(scores), 3),
                    "max_sentence": round(max(scores), 3),
                    "policy_a_refines": True,
                    "policy_b_refines": False,
                    "verdict": "A_OVER_REFINES",
                })
            elif not policy_a and policy_b:
                # Original MISSES a bad sentence (masking)
                failing = [j for j, s in enumerate(scores) if s < thr]
                disagree_b_refines_more += 1
                cases.append({
                    "record": i, "threshold": thr,
                    "agg": round(agg, 3),
                    "min_sentence": round(min(scores), 3),
                    "max_sentence": round(max(scores), 3),
                    "policy_a_refines": False,
                    "policy_b_refines": True,
                    "verdict": "A_MASKS_BAD_SENTENCE",
                    "masked_sentence_indices": str(failing),
                })
        rows.append({
            "threshold": thr,
            "total_scenarios": len([r for r in records if r.get("sentence_scores")]),
            "agree": agree,
            "disagree_a_over_refines": disagree_a_refines_more,
            "disagree_a_masks_bad_sentence": disagree_b_refines_more,
            "disagreement_pct": round(
                100 * (disagree_a_refines_more + disagree_b_refines_more) /
                max(1, len([r for r in records if r.get("sentence_scores")])), 1),
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# TEST 2 — Intra-scenario variance
# ──────────────────────────────────────────────────────────────────────────────
def test_score_variance(records):
    rows = []
    for i, r in enumerate(records):
        scores = r.get("sentence_scores", [])
        agg = r.get("score", 0) or 0
        if not scores or len(scores) < 2:
            continue
        spread = round(max(scores) - min(scores), 4)
        std = round(statistics.stdev(scores), 4)
        cv = round(std / agg * 100, 2) if agg else 0  # coefficient of variation %
        rows.append({
            "record_idx": i,
            "task": r.get("task", "?"),
            "label": r.get("label", "?"),
            "cs_type": r.get("cs_type", "?"),
            "n_sentences": len(scores),
            "agg_score": round(agg, 4),
            "min_sentence": round(min(scores), 4),
            "max_sentence": round(max(scores), 4),
            "spread": spread,
            "std_dev": std,
            "coeff_variation_pct": cv,
            "sentence_scores": str([round(s, 3) for s in scores]),
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# TEST 3 — Masking simulation (synthetic cases)
# ──────────────────────────────────────────────────────────────────────────────
def test_masking_simulation():
    """
    Construct synthetic scenario score sets that illustrate masking.
    Each case: N-1 excellent sentences + 1 bad sentence.
    Show aggregate vs per-sentence policy outcome at threshold=8.0.
    """
    THR = 8.0
    synthetic_cases = [
        # (description, sentence_scores)
        ("2 good + 1 bad",     [9.0, 9.0, 5.0]),
        ("3 good + 1 bad",     [9.5, 9.0, 9.2, 4.0]),
        ("4 good + 1 bad",     [9.0, 9.0, 9.0, 9.0, 6.0]),
        ("barely above thresh", [8.5, 8.5, 8.5, 7.9]),  # agg=8.35, min=7.9
        ("all just above",      [8.1, 8.2, 8.0, 8.1]),  # agg=8.1, all fine
        ("real data-like",      [9.5, 9.0, 7.2, 9.3]),  # one bad hidden
        ("all below thresh",    [7.5, 7.5, 7.5, 7.5]),  # both agree: refine
        ("all above thresh",    [8.5, 8.6, 8.7, 8.8]),  # both agree: accept
    ]

    rows = []
    for desc, scores in synthetic_cases:
        n = len(scores)
        agg = round(sum(scores) / n, 3)
        failing = [j for j, s in enumerate(scores) if s < THR]
        policy_a = agg < THR
        policy_b = bool(failing)
        sentences_refiner_a_sees = n if policy_a else 0
        sentences_refiner_b_sees = len(failing) if policy_b else 0
        rows.append({
            "case": desc,
            "n_sentences": n,
            "sentence_scores": str(scores),
            "agg_score": agg,
            "threshold": THR,
            "policy_a_refines": policy_a,
            "policy_b_refines": policy_b,
            "sentences_refiner_a_handles": sentences_refiner_a_sees,
            "sentences_refiner_b_handles": sentences_refiner_b_sees,
            "masked_bad_sentences": len(failing) if not policy_a and policy_b else 0,
            "verdict": (
                "A_MASKS_BAD_SENTENCE" if (not policy_a and policy_b) else
                "A_OVER_REFINES"       if (policy_a and not policy_b) else
                "BOTH_AGREE"
            ),
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# TEST 4 — Refiner targeting efficiency
# ──────────────────────────────────────────────────────────────────────────────
def test_refiner_efficiency(records, threshold=8.0):
    """
    For every record where refinement actually ran (instance_refine_counts > 0),
    compare how many sentences each policy would send to the LLM.
    """
    rows = []
    for i, r in enumerate(records):
        scores = r.get("sentence_scores", [])
        agg = r.get("score", 0) or 0
        refine_counts = r.get("instance_refine_counts", [])
        if not scores:
            continue

        actually_refined = [j for j, rc in enumerate(refine_counts) if rc and int(rc) > 0]
        failing_sentences = [j for j, s in enumerate(scores) if s < threshold]

        # Policy A: if agg < threshold, send ALL sentences to refiner
        policy_a_sends = len(scores) if agg < threshold else 0
        # Policy B: send only failing sentences
        policy_b_sends = len(failing_sentences) if failing_sentences else 0

        savings = policy_a_sends - policy_b_sends
        savings_pct = round(100 * savings / max(1, policy_a_sends), 1) if policy_a_sends > 0 else 0

        rows.append({
            "record_idx": i,
            "task": r.get("task", "?"),
            "cs_type": r.get("cs_type", "?"),
            "n_sentences": len(scores),
            "agg_score": round(agg, 3),
            "failing_sentence_count": len(failing_sentences),
            "actually_refined_count": len(actually_refined),
            "policy_a_sends_to_refiner": policy_a_sends,
            "policy_b_sends_to_refiner": policy_b_sends,
            "sentences_saved": savings,
            "savings_pct": savings_pct,
            "sentence_scores": str([round(s, 3) for s in scores]),
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Write CSV helper
# ──────────────────────────────────────────────────────────────────────────────
def write_csv(path, rows):
    if not rows:
        print(f"  [SKIP] No data for {path.name}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [OK] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────
def build_summary(records, variance_rows, masking_rows, efficiency_rows, threshold=8.0):
    scores_all = [r.get("score", 0) for r in records if r.get("sentence_scores")]
    all_sent_scores = [s for r in records for s in r.get("sentence_scores", [])]

    masking_cases = [m for m in masking_rows if m["verdict"] == "A_MASKS_BAD_SENTENCE"]
    over_refine_cases = [m for m in masking_rows if m["verdict"] == "A_OVER_REFINES"]

    avg_savings_pct = (
        round(sum(e["savings_pct"] for e in efficiency_rows) / len(efficiency_rows), 1)
        if efficiency_rows else 0
    )
    total_a_sends = sum(e["policy_a_sends_to_refiner"] for e in efficiency_rows)
    total_b_sends = sum(e["policy_b_sends_to_refiner"] for e in efficiency_rows)

    avg_spread = round(
        sum(v["spread"] for v in variance_rows) / len(variance_rows), 4
    ) if variance_rows else 0

    return [
        {"metric": "Total scenarios analysed",                       "value": len([r for r in records if r.get("sentence_scores")]),  "interpretation": ""},
        {"metric": "Total sentences analysed",                       "value": len(all_sent_scores),                                  "interpretation": ""},
        {"metric": "Avg intra-scenario score spread (max-min)",      "value": avg_spread,                                            "interpretation": "Higher = aggregate is more misleading"},
        {"metric": f"Masking cases at thr={threshold} (synthetic)",  "value": len(masking_cases),                                    "interpretation": "Scenarios where Policy-A misses a bad sentence"},
        {"metric": f"Over-refine cases at thr={threshold} (synth)",  "value": len(over_refine_cases),                                "interpretation": "Scenarios where Policy-A refines unnecessarily"},
        {"metric": "Total LLM calls to Refiner — Policy A (orig)",   "value": total_a_sends,                                        "interpretation": ""},
        {"metric": "Total LLM calls to Refiner — Policy B (mod)",    "value": total_b_sends,                                        "interpretation": ""},
        {"metric": "LLM call savings vs original (%)",               "value": f"{round(100*(total_a_sends - total_b_sends)/max(1,total_a_sends),1)}%", "interpretation": "Per-sentence is more efficient"},
        {"metric": "Avg savings per scenario (%)",                   "value": f"{avg_savings_pct}%",                                "interpretation": ""},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f"[per-sent-test] Loading {JSONL_B}")
    records = load_jsonl(JSONL_B)
    print(f"[per-sent-test] {len(records)} records loaded\n")

    print("Running Test 1 — Policy disagreement...")
    t1 = test_policy_disagreement(records, THRESHOLDS)
    write_csv(OUT_DIR / "policy_disagreement.csv", t1)
    print()

    print("Running Test 2 — Intra-scenario score variance...")
    t2 = test_score_variance(records)
    write_csv(OUT_DIR / "score_variance.csv", t2)
    print()

    print("Running Test 3 — Masking simulation (synthetic)...")
    t3 = test_masking_simulation()
    write_csv(OUT_DIR / "masking_simulation.csv", t3)
    print()

    print("Running Test 4 — Refiner targeting efficiency...")
    t4 = test_refiner_efficiency(records, threshold=8.0)
    write_csv(OUT_DIR / "refiner_efficiency.csv", t4)
    print()

    print("Building summary table...")
    summary = build_summary(records, t2, t3, t4, threshold=8.0)
    write_csv(OUT_DIR / "per_sentence_proof_summary.csv", summary)
    print()

    # ── Pretty-print key findings ──────────────────────────────────────────
    print("=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print("\n[Test 1] Policy disagreement at each threshold:")
    print(f"  {'Threshold':>9} | {'Agree':>5} | {'A masks bad':>11} | {'A over-refines':>14} | {'Disagree%':>9}")
    print("  " + "-" * 58)
    for row in t1:
        print(f"  {row['threshold']:>9} | {row['agree']:>5} | {row['disagree_a_masks_bad_sentence']:>11} | {row['disagree_a_over_refines']:>14} | {row['disagreement_pct']:>8}%")

    print("\n[Test 2] Intra-scenario score variance:")
    for row in t2:
        print(f"  Record {row['record_idx']}: agg={row['agg_score']}  spread={row['spread']}  std={row['std_dev']}  CV={row['coeff_variation_pct']}%  scores={row['sentence_scores']}")

    print("\n[Test 3] Masking simulation (synthetic, threshold=8.0):")
    print(f"  {'Case':<25} | {'Agg':>5} | {'PolicyA':>7} | {'PolicyB':>7} | {'Verdict'}")
    print("  " + "-" * 75)
    for row in t3:
        print(f"  {row['case']:<25} | {row['agg_score']:>5} | {str(row['policy_a_refines']):>7} | {str(row['policy_b_refines']):>7} | {row['verdict']}")

    print("\n[Test 4] Refiner efficiency (actual data, threshold=8.0):")
    total_a = sum(r["policy_a_sends_to_refiner"] for r in t4)
    total_b = sum(r["policy_b_sends_to_refiner"] for r in t4)
    print(f"  Policy A (original) would send {total_a} sentences to Refiner across {len(t4)} scenarios")
    print(f"  Policy B (modified) sends      {total_b} sentences to Refiner across {len(t4)} scenarios")
    if total_a > 0:
        print(f"  Saving: {total_a - total_b} unnecessary LLM calls ({round(100*(total_a-total_b)/total_a,1)}% reduction)")
    print()

    print("[per-sent-test] Done. CSVs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
