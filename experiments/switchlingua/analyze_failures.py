"""
analyze_failures.py
====================
Thesis Experiment — Failure Case Analysis

Identifies and categorises failure cases across all three systems:

Failure categories:
  low_overall        → scenario-level score < LOW_SCORE_THRESHOLD (default 3.0)
  low_fluency        → any sentence fluency score < AGENT_THRESHOLD
  low_naturalness    → any sentence naturalness score < AGENT_THRESHOLD
  low_cs_ratio       → CS ratio score < AGENT_THRESHOLD (ratio mismatch)
  validation_fail    → task_validation_result indicates failure
  empty_output       → data_generation_result is empty or missing
  det_cs_mismatch    → System B: deterministic AR% outside ±20pp of configured ratio

Writes to:
  experiments/outputs/switchlingua/
    failure_cases.csv           ← one row per failing scenario, with category tags
    failure_summary.csv         ← count per failure category per system

Usage:
    python experiments/switchlingua/analyze_failures.py [--systems a b c]
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

from utils import compute_true_cs_stats  # noqa: E402

OUT_ROOT = ROOT / "experiments" / "outputs" / "switchlingua"
JSONL = {
    "System_A": OUT_ROOT / "system_a" / "Arabic.jsonl",
    "System_B": OUT_ROOT / "system_b" / "Arabic.jsonl",
    "System_C": OUT_ROOT / "system_c" / "Arabic.jsonl",
}

LOW_SCORE_THRESHOLD = 3.0
AGENT_THRESHOLD     = 3.0
CS_MISMATCH_MARGIN  = 20.0  # percentage points


def _iter_jsonl(path: pathlib.Path):
    if not path.exists():
        print(f"[failures] MISSING: {path}")
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _extract_scalar(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for k in ("score", "ratio_score", "task_score"):
            sv = v.get(k)
            if isinstance(sv, (int, float)):
                return float(sv)
    return None


def _is_validation_fail(rec: dict) -> bool:
    tv = rec.get("task_validation_result") or {}
    if isinstance(tv, dict):
        passed = tv.get("passed", tv.get("task_score", 5.0))
        if isinstance(passed, bool):
            return not passed
        if isinstance(passed, (int, float)):
            return passed < AGENT_THRESHOLD
    return False


def analyse_record(rec: dict, system: str) -> dict | None:
    """Return a failure row if the record has any failure, else None."""
    sentences = rec.get("data_generation_result", [])
    overall = _extract_scalar(rec.get("score")) or _extract_scalar(rec.get("weighted_score"))

    categories = []

    if not sentences:
        categories.append("empty_output")

    if overall is not None and overall < LOW_SCORE_THRESHOLD:
        categories.append("low_overall")

    if _is_validation_fail(rec):
        categories.append("validation_fail")

    # Per-sentence agent failures (System B has these; A/C do not)
    fluency_per = rec.get("fluency_results_per_instances", [])
    nat_per     = rec.get("naturalness_results_per_instances", [])
    cs_per      = rec.get("cs_ratio_results_per_instances", [])

    for i in range(len(sentences)):
        fs = _extract_scalar(fluency_per[i]) if i < len(fluency_per) else None
        if fs is not None and fs < AGENT_THRESHOLD:
            categories.append("low_fluency")
            break
    for i in range(len(sentences)):
        ns = _extract_scalar(nat_per[i]) if i < len(nat_per) else None
        if ns is not None and ns < AGENT_THRESHOLD:
            categories.append("low_naturalness")
            break
    for i in range(len(sentences)):
        cs = _extract_scalar(cs_per[i]) if i < len(cs_per) else None
        if cs is not None and cs < AGENT_THRESHOLD:
            categories.append("low_cs_ratio")
            break

    # Deterministic CS mismatch (works for all systems post-hoc)
    configured_ratio_str = str(rec.get("cs_ratio", "50_50"))
    try:
        # Parse "70_30" or "70" style
        parts = configured_ratio_str.replace(":", "_").split("_")
        configured_ar = float(parts[0]) if parts else 50.0
    except (ValueError, IndexError):
        configured_ar = 50.0

    if sentences:
        det_ars = [compute_true_cs_stats(s).get("ar_pct", 0) for s in sentences]
        mean_ar = statistics.mean(det_ars)
        if abs(mean_ar - configured_ar) > CS_MISMATCH_MARGIN:
            categories.append("det_cs_mismatch")

    if not categories:
        return None

    return {
        "system": system,
        "task": rec.get("task", ""),
        "label": rec.get("label", rec.get("topic", "")),
        "overall_score": overall,
        "n_sentences": len(sentences),
        "configured_cs_ratio": configured_ratio_str,
        "failure_categories": "|".join(sorted(set(categories))),
        "n_categories": len(set(categories)),
    }


def _write_csv(path: pathlib.Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[failures] {path}")


SYSTEM_MAP = {"a": "System_A", "b": "System_B", "c": "System_C"}
ALL_CATEGORIES = ["empty_output", "low_overall", "validation_fail",
                  "low_fluency", "low_naturalness", "low_cs_ratio", "det_cs_mismatch"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse failure cases across systems.")
    parser.add_argument("--systems", nargs="+", choices=["a", "b", "c"], default=["a", "b", "c"])
    args = parser.parse_args()

    all_failures: list[dict] = []
    totals: dict[str, int] = {}

    for key in args.systems:
        label = SYSTEM_MAP[key]
        system_total = 0
        for rec in _iter_jsonl(JSONL[label]):
            system_total += 1
            fail_row = analyse_record(rec, label)
            if fail_row:
                all_failures.append(fail_row)
        totals[label] = system_total
        sys_fails = [r for r in all_failures if r["system"] == label]
        print(f"[failures] {label}: {len(sys_fails)}/{system_total} scenarios have failures")

    _write_csv(OUT_ROOT / "failure_cases.csv", all_failures)

    # Summary per system + category
    summary_rows = []
    for key in args.systems:
        label = SYSTEM_MAP[key]
        total = totals.get(label, 0)
        for cat in ALL_CATEGORIES:
            count = sum(1 for r in all_failures
                        if r["system"] == label and cat in r["failure_categories"].split("|"))
            summary_rows.append({
                "system": label,
                "failure_category": cat,
                "count": count,
                "total_scenarios": total,
                "failure_rate_pct": round(count / total * 100, 1) if total else 0,
            })
    _write_csv(OUT_ROOT / "failure_summary.csv", summary_rows)
    print("[failures] Done.")
