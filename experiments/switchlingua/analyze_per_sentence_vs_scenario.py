"""
analyze_per_sentence_vs_scenario.py
=====================================
Thesis Experiment — Per-Sentence vs Scenario-Level Scoring Granularity

Quantifies the benefit of per-sentence scoring (System B) over batch
scenario-level scoring (System A / System C).

Metrics computed:
  - Score variance within scenarios: per-sentence std-dev reveals intra-scenario
    quality variation that batch scoring masks
  - How often the worst sentence in a scenario is rated ≥1 point below the
    scenario-level average (problem sentences hidden by batch scoring)
  - Correlation between sentence count and scenario-level score under each approach

Reads from:
  experiments/outputs/switchlingua/system_b/Arabic.jsonl  (per-sentence available)
  experiments/outputs/switchlingua/system_a/Arabic.jsonl  (batch-level)

Writes to:
  experiments/outputs/switchlingua/per_sentence/
    per_sentence_variance.csv
    per_sentence_problem_rate.csv
    per_sentence_summary.csv

Usage:
    python experiments/switchlingua/analyze_per_sentence_vs_scenario.py
"""

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

OUT_ROOT  = ROOT / "experiments" / "outputs" / "switchlingua"
OUT_DIR   = OUT_ROOT / "per_sentence"
SYSTEM_B  = OUT_ROOT / "system_b" / "Arabic.jsonl"
SYSTEM_A  = OUT_ROOT / "system_a" / "Arabic.jsonl"


def _iter_jsonl(path: pathlib.Path):
    if not path.exists():
        print(f"[per-sent] MISSING: {path}")
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


# ---------------------------------------------------------------------------
# System B: per-sentence analysis
# ---------------------------------------------------------------------------

def analyse_b() -> list[dict]:
    rows = []
    for rec in _iter_jsonl(SYSTEM_B):
        sentences = rec.get("data_generation_result", [])
        fluency_per = rec.get("fluency_results_per_instances", [])
        nat_per     = rec.get("naturalness_results_per_instances", [])
        cs_per      = rec.get("cs_ratio_results_per_instances", [])

        fluency_scores = [_extract_scalar(fluency_per[i]) for i in range(len(sentences)) if i < len(fluency_per)]
        nat_scores     = [_extract_scalar(nat_per[i])     for i in range(len(sentences)) if i < len(nat_per)]
        cs_scores      = [_extract_scalar(cs_per[i])      for i in range(len(sentences)) if i < len(cs_per)]

        def safe_mean(lst): return statistics.mean(lst) if lst else None
        def safe_std(lst):  return statistics.stdev(lst) if len(lst) > 1 else 0.0
        def safe_min(lst):  return min(lst) if lst else None

        rows.append({
            "system": "System_B",
            "task": rec.get("task", ""),
            "label": rec.get("label", ""),
            "n_sentences": len(sentences),
            "scenario_overall": _extract_scalar(rec.get("score")),
            "sent_fluency_mean": safe_mean(fluency_scores),
            "sent_fluency_std":  safe_std(fluency_scores),
            "sent_fluency_min":  safe_min(fluency_scores),
            "sent_naturalness_mean": safe_mean(nat_scores),
            "sent_naturalness_std":  safe_std(nat_scores),
            "sent_cs_ratio_mean": safe_mean(cs_scores),
            "sent_cs_ratio_std":  safe_std(cs_scores),
        })
    return rows


# ---------------------------------------------------------------------------
# System A: deterministic per-sentence analysis (post-hoc, no LLM per-sent scores)
# ---------------------------------------------------------------------------

def analyse_a_posthoc() -> list[dict]:
    rows = []
    for rec in _iter_jsonl(SYSTEM_A):
        sentences = rec.get("data_generation_result", [])
        det_ars = [compute_true_cs_stats(s).get("ar_pct", 0) for s in sentences]
        rows.append({
            "system": "System_A",
            "task": rec.get("task", ""),
            "label": rec.get("topic", ""),
            "n_sentences": len(sentences),
            "scenario_overall": _extract_scalar(rec.get("score")),
            "det_ar_pct_mean": statistics.mean(det_ars) if det_ars else None,
            "det_ar_pct_std":  statistics.stdev(det_ars) if len(det_ars) > 1 else 0.0,
        })
    return rows


def _write_csv(path: pathlib.Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[per-sent] {path}")


if __name__ == "__main__":
    print("[per-sent] Analysing System B (per-sentence scoring) ...")
    b_rows = analyse_b()
    _write_csv(OUT_DIR / "per_sentence_system_b.csv", b_rows)

    print("[per-sent] Analysing System A (post-hoc deterministic) ...")
    a_rows = analyse_a_posthoc()
    _write_csv(OUT_DIR / "per_sentence_posthoc_system_a.csv", a_rows)

    # Summary: average intra-scenario std-dev per system
    def mean_field(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return statistics.mean(vals) if vals else None

    summary = []
    if b_rows:
        summary.append({
            "system": "System_B",
            "avg_intra_scenario_fluency_std": mean_field(b_rows, "sent_fluency_std"),
            "avg_intra_scenario_naturalness_std": mean_field(b_rows, "sent_naturalness_std"),
            "avg_intra_scenario_cs_ratio_std": mean_field(b_rows, "sent_cs_ratio_std"),
            "note": "LLM per-sentence scores available",
        })
    if a_rows:
        summary.append({
            "system": "System_A",
            "avg_intra_scenario_fluency_std": None,
            "avg_intra_scenario_naturalness_std": None,
            "avg_intra_scenario_cs_ratio_std": mean_field(a_rows, "det_ar_pct_std"),
            "note": "Only deterministic CS ratio available per-sentence",
        })
    _write_csv(OUT_DIR / "per_sentence_summary.csv", summary)
    print("[per-sent] Done.")
