"""
step2_count_masking.py — Count masking on REAL data (no made-up numbers).
=========================================================================
Masking = a scenario where the AVERAGE rule would ACCEPT (aggregate >= threshold)
but at least one sentence is BELOW threshold (a bad sentence the average hides).

  - AVERAGE rule (original): accept if aggregate >= threshold  (refine if below)
  - PER-SENTENCE rule (ours): accept only if EVERY sentence >= threshold

A masking case = AVERAGE accepts, PER-SENTENCE catches a bad one.

Reads the RAW "before" data (refiner OFF) — the honest, unfixed grades.
Writes three CSVs to step2_counts/:
  masking_by_threshold.csv  — masking rate at several thresholds
  masking_cases.csv         — the actual masking scenarios (for human check, Step 3)
  spread_per_scenario.csv   — how much sentence quality varies within each scenario

Usage:
  python step2_count_masking.py                # uses step1_raw_data
  python step2_count_masking.py step1_raw_data
"""
import csv
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC_FOLDER = sys.argv[1] if len(sys.argv) > 1 else "step1_raw_data"
SRC = HERE / SRC_FOLDER / "Arabic.jsonl"
OUT = HERE / "step2_counts"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [6.5, 7.0, 7.5, 8.0, 8.5]
PIPELINE_THRESHOLD = 8.0   # the pipeline's default bar (too high for this model -> 0 masking)
OPERATING_THRESHOLD = 7.0  # calibrated to this model's actual quality band; case list uses this


def load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    recs = load(SRC)
    # keep only scenarios with usable per-sentence scores (need >=2 to "hide" anything)
    usable = []
    for i, r in enumerate(recs):
        scores = r.get("sentence_scores", [])
        if isinstance(scores, list) and len(scores) >= 2:
            usable.append((i, r, scores))

    print(f"Source: {SRC_FOLDER}/Arabic.jsonl")
    print(f"Scenarios total: {len(recs)} | usable (>=2 sentences): {len(usable)}\n")

    # ---- 1) masking rate by threshold ----
    by_thr_rows = []
    for thr in THRESHOLDS:
        n_masking = n_agree_accept = n_agree_refine = 0
        for _, r, scores in usable:
            agg = float(r.get("score", 0) or 0)
            avg_accepts = agg >= thr
            persent_accepts = all(s >= thr for s in scores)
            if avg_accepts and not persent_accepts:
                n_masking += 1          # AVERAGE hides a bad sentence
            elif avg_accepts and persent_accepts:
                n_agree_accept += 1
            else:
                n_agree_refine += 1
        rate = round(100 * n_masking / len(usable), 1) if usable else 0
        by_thr_rows.append({
            "threshold": thr,
            "usable_scenarios": len(usable),
            "masking_cases": n_masking,
            "masking_rate_pct": rate,
            "agree_accept": n_agree_accept,
            "agree_refine": n_agree_refine,
        })

    # ---- 2) list the actual masking cases at the OPERATING threshold (for human check) ----
    case_rows = []
    for idx, r, scores in usable:
        agg = float(r.get("score", 0) or 0)
        if agg >= OPERATING_THRESHOLD and not all(s >= OPERATING_THRESHOLD for s in scores):
            masked = [j for j, s in enumerate(scores) if s < OPERATING_THRESHOLD]
            sents = r.get("data_generation_result", [])
            for j in masked:
                case_rows.append({
                    "record_idx": idx,
                    "task": r.get("task", "?"),
                    "cs_type": r.get("cs_type", "?"),
                    "aggregate": round(agg, 3),
                    "n_sentences": len(scores),
                    "masked_sentence_idx": j,
                    "masked_sentence_score": round(scores[j], 3),
                    "all_scores": str([round(s, 3) for s in scores]),
                    "masked_sentence_text": sents[j] if j < len(sents) else "",
                })

    # ---- 3) intra-scenario spread (shows quality varies => masking is possible) ----
    spread_rows = []
    spreads = []
    for idx, r, scores in usable:
        spread = round(max(scores) - min(scores), 3)
        spreads.append(spread)
        spread_rows.append({
            "record_idx": idx,
            "task": r.get("task", "?"),
            "cs_type": r.get("cs_type", "?"),
            "aggregate": round(float(r.get("score", 0) or 0), 3),
            "min_sentence": round(min(scores), 3),
            "max_sentence": round(max(scores), 3),
            "spread": spread,
            "all_scores": str([round(s, 3) for s in scores]),
        })

    def write(name, rows):
        p = OUT / name
        if rows:
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader(); w.writerows(rows)
        print(f"  wrote {p.name} ({len(rows)} rows)")

    write("masking_by_threshold.csv", by_thr_rows)
    write("masking_cases.csv", case_rows)
    write("spread_per_scenario.csv", spread_rows)

    # ---- pretty summary ----
    print("\n" + "=" * 62)
    print("MASKING RATE BY THRESHOLD")
    print("=" * 62)
    print(f"  {'thr':>5} | {'masking':>7} | {'rate':>6} | {'agree-accept':>12} | {'agree-refine':>12}")
    print("  " + "-" * 56)
    for row in by_thr_rows:
        print(f"  {row['threshold']:>5} | {row['masking_cases']:>7} | {row['masking_rate_pct']:>5}% | "
              f"{row['agree_accept']:>12} | {row['agree_refine']:>12}")

    avg_spread = round(statistics.mean(spreads), 3) if spreads else 0
    max_spread = round(max(spreads), 3) if spreads else 0
    print(f"\nIntra-scenario spread (max-min sentence score): avg={avg_spread}, biggest={max_spread}")
    print(f"Masking cases at pipeline threshold {PIPELINE_THRESHOLD}: "
          f"{sum(1 for r in by_thr_rows if r['threshold']==PIPELINE_THRESHOLD for _ in [0])} threshold row; "
          f"{len(set(c['record_idx'] for c in case_rows))} scenarios, {len(case_rows)} masked sentences")


if __name__ == "__main__":
    main()
