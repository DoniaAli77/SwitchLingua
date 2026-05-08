"""
score_ablation.py
=================
Thesis Experiment — Score Ablation Results

Loads JSONL outputs from run_ablation.py (one file per ablation condition)
and computes a score comparison table showing the contribution of each
removed component relative to the full system.

Reads from:
  experiments/outputs/switchlingua/ablation/<condition>/Arabic.jsonl

Writes to:
  experiments/outputs/switchlingua/ablation/
    ablation_scores.csv         ← per-scenario per-condition
    ablation_summary.csv        ← mean scores per condition
    ablation_delta.csv          ← delta from "full" condition

Usage:
    python experiments/switchlingua/score_ablation.py [--conditions full no_tv ...]
"""

import argparse
import csv
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_ROOT = ROOT / "experiments" / "outputs" / "switchlingua" / "ablation"

ALL_CONDITIONS = ["full", "no_tv", "no_reeval", "no_det_cs", "no_refine", "no_per_sent"]

AGENT_SCORE_KEYS = {
    "fluency":         ["fluency_result", "fluency_score"],
    "naturalness":     ["naturalness_result", "naturalness_score"],
    "social_cultural": ["social_cultural_result", "social_cultural_score"],
    "cs_ratio":        ["cs_ratio_result", "cs_ratio_score"],
    "task_validation": ["task_validation_result", "task_score"],
}


def _iter_jsonl(path: pathlib.Path):
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _extract_scalar(record: dict, keys: list[str]):
    for k in keys:
        v = record.get(k)
        if v is None:
            continue
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict):
            for sub in ("score", "ratio_score", "task_score"):
                sv = v.get(sub)
                if isinstance(sv, (int, float)):
                    return float(sv)
    return None


def load_condition(condition: str) -> list[dict]:
    path = OUT_ROOT / condition / "Arabic.jsonl"
    rows = []
    for rec in _iter_jsonl(path):
        row = {
            "condition": condition,
            "task": rec.get("task", ""),
            "label": rec.get("label", rec.get("topic", "")),
            "overall": _extract_scalar(rec, ["score", "weighted_score"]),
        }
        for agent, keys in AGENT_SCORE_KEYS.items():
            row[agent] = _extract_scalar(rec, keys)
        rows.append(row)
    return rows


def summarise(all_rows: list[dict]) -> list[dict]:
    by_condition: dict[str, list] = {}
    for r in all_rows:
        by_condition.setdefault(r["condition"], []).append(r)

    summary = []
    for cond, rows in sorted(by_condition.items()):
        def mean(key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return statistics.mean(vals) if vals else None
        row = {"condition": cond, "n": len(rows), "avg_overall": mean("overall")}
        for agent in AGENT_SCORE_KEYS:
            row[f"avg_{agent}"] = mean(agent)
        summary.append(row)
    return summary


def compute_delta(summary: list[dict]) -> list[dict]:
    full_row = next((r for r in summary if r["condition"] == "full"), None)
    if full_row is None:
        return []
    delta = []
    for row in summary:
        d = {"condition": row["condition"], "n": row["n"]}
        for k, v in row.items():
            if k in ("condition", "n"):
                continue
            full_v = full_row.get(k)
            d[f"delta_{k}"] = (v - full_v) if (v is not None and full_v is not None) else None
        delta.append(d)
    return delta


def _write_csv(path: pathlib.Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ablation-score] {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score ablation study outputs.")
    parser.add_argument("--conditions", nargs="+", choices=ALL_CONDITIONS, default=ALL_CONDITIONS)
    args = parser.parse_args()

    all_rows: list[dict] = []
    for cond in args.conditions:
        rows = load_condition(cond)
        print(f"[ablation-score] {cond}: {len(rows)} records")
        all_rows.extend(rows)

    if not all_rows:
        print("[ablation-score] No records. Run run_ablation.py first.")
        sys.exit(1)

    _write_csv(OUT_ROOT / "ablation_scores.csv", all_rows)

    summary = summarise(all_rows)
    _write_csv(OUT_ROOT / "ablation_summary.csv", summary)

    delta = compute_delta(summary)
    _write_csv(OUT_ROOT / "ablation_delta.csv", delta)

    print("\n[ablation-score] Summary:")
    for row in summary:
        print(f"  {row['condition']:20s}  avg_overall={row['avg_overall']}")
