"""
analyze_cost_quality.py
========================
Thesis Experiment — API Cost vs Quality Analysis

Estimates token usage and API cost per system and correlates cost per scenario
with the quality scores produced.

Cost assumptions (update these if pricing changes):
  GPT-4o      input: $0.005 / 1K tokens   output: $0.015 / 1K tokens
  GPT-4o-mini input: $0.00015 / 1K tokens  output: $0.0006 / 1K tokens

Token counts are read from:
  - `usage` key in JSONL records (if the pipeline logged it)
  - Estimated from text lengths as fallback (rough: 1 token ≈ 4 chars)

Writes to:
  experiments/outputs/switchlingua/
    cost_quality_per_scenario.csv
    cost_quality_summary.csv

Usage:
    python experiments/switchlingua/analyze_cost_quality.py [--systems a b c]
"""

import argparse
import csv
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_ROOT = ROOT / "experiments" / "outputs" / "switchlingua"
JSONL = {
    "System_A": (OUT_ROOT / "system_a_original_gpt4o"  / "Arabic.jsonl", "gpt-4o"),
    "System_B": (OUT_ROOT / "system_b_modified_mini"   / "Arabic.jsonl", "gpt-4o-mini"),
    "System_C": (OUT_ROOT / "system_c_original_mini"   / "Arabic.jsonl", "gpt-4o-mini"),
}

PRICING = {
    "gpt-4o":      {"input": 0.005 / 1000, "output": 0.015 / 1000},
    "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
}


def _iter_jsonl(path: pathlib.Path):
    if not path.exists():
        print(f"[cost] MISSING: {path}")
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars."""
    return max(1, len(text) // 4)


def _extract_scalar(rec: dict, keys: list[str]):
    for k in keys:
        v = rec.get(k)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict):
            for s in ("score", "ratio_score", "task_score"):
                sv = v.get(s)
                if isinstance(sv, (int, float)):
                    return float(sv)
    return None


def analyse(system: str, jsonl_path: pathlib.Path, model: str) -> list[dict]:
    pricing = PRICING[model]
    rows = []
    for rec in _iter_jsonl(jsonl_path):
        usage = rec.get("usage") or {}
        prompt_tokens    = usage.get("prompt_tokens") or _estimate_tokens(json.dumps(rec))
        completion_tokens = usage.get("completion_tokens") or _estimate_tokens(
            " ".join(rec.get("data_generation_result", [])))

        cost = (prompt_tokens * pricing["input"]) + (completion_tokens * pricing["output"])
        overall = _extract_scalar(rec, ["score", "weighted_score"])

        rows.append({
            "system": system,
            "model": model,
            "task": rec.get("task", ""),
            "label": rec.get("label", rec.get("topic", "")),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost_usd": round(cost, 6),
            "overall_score": overall,
            "n_sentences": len(rec.get("data_generation_result", [])),
            "usage_source": "logged" if usage else "estimated",
        })
    return rows


def summarise(all_rows: list[dict]) -> list[dict]:
    by_system: dict[str, list] = {}
    for r in all_rows:
        by_system.setdefault(r["system"], []).append(r)
    summary = []
    for sys_name, rows in sorted(by_system.items()):
        def mean(key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return round(statistics.mean(vals), 4) if vals else None
        summary.append({
            "system": sys_name,
            "model": rows[0]["model"],
            "n_scenarios": len(rows),
            "avg_total_tokens": mean("total_tokens"),
            "avg_cost_usd": mean("estimated_cost_usd"),
            "total_cost_usd": round(sum(r["estimated_cost_usd"] for r in rows), 4),
            "avg_overall_score": mean("overall_score"),
            "cost_per_score_point": (
                round(sum(r["estimated_cost_usd"] for r in rows) /
                      sum(r["overall_score"] for r in rows if r["overall_score"]), 6)
                if any(r["overall_score"] for r in rows) else None
            ),
        })
    return summary


def _write_csv(path: pathlib.Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[cost] {path}")


SYSTEM_MAP = {"a": "System_A", "b": "System_B", "c": "System_C"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API cost vs quality analysis.")
    parser.add_argument("--systems", nargs="+", choices=["a", "b", "c"], default=["a", "b", "c"])
    args = parser.parse_args()

    all_rows: list[dict] = []
    for key in args.systems:
        label = SYSTEM_MAP[key]
        path, model = JSONL[label]
        rows = analyse(label, path, model)
        print(f"[cost] {label}: {len(rows)} records")
        all_rows.extend(rows)

    if not all_rows:
        print("[cost] No records. Run run_full_pipeline_generation.py first.")
        sys.exit(1)

    _write_csv(OUT_ROOT / "cost_quality_per_scenario.csv", all_rows)
    summary = summarise(all_rows)
    _write_csv(OUT_ROOT / "cost_quality_summary.csv", summary)

    print("\n[cost] Summary:")
    for row in summary:
        print(f"  {row['system']:10s} {row['model']:15s} "
              f"avg_score={row['avg_overall_score']}  avg_cost=${row['avg_cost_usd']}")
