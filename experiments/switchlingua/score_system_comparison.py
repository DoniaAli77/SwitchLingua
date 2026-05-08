"""
score_system_comparison.py
===========================
Thesis Experiment — System A / B / C Score Comparison

Loads JSONL outputs from all three systems and computes comparison tables
covering: overall score, per-agent scores, CS ratio accuracy, sentence
quality distribution.

Reads from:
  experiments/outputs/switchlingua/system_a/Arabic.jsonl
  experiments/outputs/switchlingua/system_b/Arabic.jsonl
  experiments/outputs/switchlingua/system_c/Arabic.jsonl

Writes to:
  experiments/outputs/switchlingua/
    system_comparison_overall.csv
    system_comparison_per_agent.csv
    system_comparison_per_task.csv

Usage:
    python experiments/switchlingua/score_system_comparison.py [--systems a b c]
"""

import argparse
import csv
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
if str(MODIFIED_CORE) not in sys.path:
    sys.path.insert(0, str(MODIFIED_CORE))

from utils import compute_true_cs_stats, weighting_scheme  # noqa: E402

OUT_ROOT = ROOT / "experiments" / "outputs" / "switchlingua"
JSONL = {
    "System_A": OUT_ROOT / "system_a" / "Arabic.jsonl",
    "System_B": OUT_ROOT / "system_b" / "Arabic.jsonl",
    "System_C": OUT_ROOT / "system_c" / "Arabic.jsonl",
}

# Agent score keys shared across systems (best-effort — fall back to None)
AGENT_SCORE_KEYS = {
    "fluency":         ["fluency_result", "fluency_score"],
    "naturalness":     ["naturalness_result", "naturalness_score"],
    "social_cultural": ["social_cultural_result", "social_cultural_score"],
    "cs_ratio":        ["cs_ratio_result", "cs_ratio_score"],
    "task_validation": ["task_validation_result", "task_score"],
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _iter_jsonl(path: pathlib.Path):
    if not path.exists():
        print(f"[score] MISSING: {path}")
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


def load_system(label: str, path: pathlib.Path) -> list[dict]:
    records = []
    for rec in _iter_jsonl(path):
        row = {"system": label, "task": rec.get("task", ""), "label": rec.get("label", rec.get("topic", ""))}
        overall = _extract_scalar(rec, ["score", "weighted_score"])
        row["overall_score"] = overall
        for agent, keys in AGENT_SCORE_KEYS.items():
            row[agent] = _extract_scalar(rec, keys)
        # CS ratio accuracy: compare configured ratio vs deterministic
        sentences = rec.get("data_generation_result", [])
        if sentences:
            det_ars = [compute_true_cs_stats(s).get("ar_pct", 0) for s in sentences]
            row["det_ar_mean"] = statistics.mean(det_ars)
        else:
            row["det_ar_mean"] = None
        records.append(row)
    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(records: list[dict]) -> dict:
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return statistics.mean(vals) if vals else None

    agents = list(AGENT_SCORE_KEYS.keys())
    row = {
        "system": records[0]["system"] if records else "?",
        "n_scenarios": len(records),
        "avg_overall": mean([r["overall_score"] for r in records]),
        "avg_det_ar_pct": mean([r["det_ar_mean"] for r in records]),
    }
    for a in agents:
        row[f"avg_{a}"] = mean([r[a] for r in records])
    return row


def per_task(records: list[dict]) -> list[dict]:
    by_task: dict[tuple, list] = {}
    for r in records:
        key = (r["system"], r["task"])
        by_task.setdefault(key, []).append(r)
    rows = []
    for (sys, task), recs in sorted(by_task.items()):
        vals = [r["overall_score"] for r in recs if r["overall_score"] is not None]
        rows.append({
            "system": sys,
            "task": task,
            "n": len(recs),
            "avg_overall": statistics.mean(vals) if vals else None,
        })
    return rows


# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

def _write_csv(path: pathlib.Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[score] {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SYSTEM_MAP = {"a": "System_A", "b": "System_B", "c": "System_C"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score and compare systems A, B, C.")
    parser.add_argument("--systems", nargs="+", choices=["a", "b", "c"], default=["a", "b", "c"])
    args = parser.parse_args()

    all_records: list[dict] = []
    for key in args.systems:
        label = SYSTEM_MAP[key]
        recs = load_system(label, JSONL[label])
        print(f"[score] Loaded {len(recs)} records for {label}")
        all_records.extend(recs)

    if not all_records:
        print("[score] No records loaded. Run run_full_pipeline_generation.py first.")
        sys.exit(1)

    # Overall aggregate
    by_system: dict[str, list] = {}
    for r in all_records:
        by_system.setdefault(r["system"], []).append(r)
    overall = [aggregate(recs) for recs in by_system.values()]
    _write_csv(OUT_ROOT / "system_comparison_overall.csv", overall)

    # Per-agent detail
    _write_csv(OUT_ROOT / "system_comparison_per_agent.csv", all_records)

    # Per-task breakdown
    _write_csv(OUT_ROOT / "system_comparison_per_task.csv", per_task(all_records))

    print("[score] Done.")
