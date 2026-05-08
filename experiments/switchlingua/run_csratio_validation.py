"""
run_csratio_validation.py
=========================
Thesis Experiment — CS Ratio Validation

For each sentence in System B outputs (Modified_Version), compare:
  - deterministic CS ratio computed by `compute_true_cs_stats()`
  - LLM-reported CS ratio from `cs_ratio_results_per_instances[i]`

For System A / System C outputs (Original_baseLine), the LLM-reported
`cs_ratio_result` is batch-level and does not have a per-sentence breakdown;
this script computes the deterministic ratio per sentence for those systems
as a post-hoc analysis.

Output:
  experiments/outputs/switchlingua/csratio/
    cs_ratio_validation_B.csv     ← system B: deterministic vs LLM per sentence
    cs_ratio_validation_A.csv     ← system A: deterministic post-hoc per sentence
    cs_ratio_validation_C.csv     ← system C: deterministic post-hoc per sentence
    cs_ratio_summary.csv          ← aggregate stats per system

Usage:
    python experiments/switchlingua/run_csratio_validation.py [--systems a b c]
"""

import argparse
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
if str(MODIFIED_CORE) not in sys.path:
    sys.path.insert(0, str(MODIFIED_CORE))

from utils import compute_true_cs_stats  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_ROOT = ROOT / "experiments" / "outputs" / "switchlingua"
SYSTEMS = {
    "a": OUT_ROOT / "system_a" / "Arabic.jsonl",
    "b": OUT_ROOT / "system_b" / "Arabic.jsonl",
    "c": OUT_ROOT / "system_c" / "Arabic.jsonl",
}
CSRATIO_OUT = OUT_ROOT / "csratio"


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse_system_b(jsonl_path: pathlib.Path, out_csv: pathlib.Path):
    """System B has per-instance LLM ratios — compare with deterministic."""
    rows = []
    for line_no, line in enumerate(_iter_jsonl(jsonl_path)):
        sentences = line.get("data_generation_result", [])
        llm_ratios = line.get("cs_ratio_results_per_instances", [])
        for i, sentence in enumerate(sentences):
            det = compute_true_cs_stats(sentence)
            llm_entry = llm_ratios[i] if i < len(llm_ratios) else {}
            rows.append({
                "line": line_no + 1,
                "sentence_index": i,
                "task": line.get("task", ""),
                "label": line.get("label", ""),
                "sentence": sentence,
                "det_ar_pct": det.get("ar_pct"),
                "det_en_pct": det.get("en_pct"),
                "det_total_tokens": det.get("total_tokens"),
                "llm_ratio_score": llm_entry.get("ratio_score"),
                "llm_computed_ratio": llm_entry.get("computed_ratio"),
                "llm_notes": llm_entry.get("notes"),
                "configured_cs_ratio": line.get("cs_ratio", ""),
            })
    _write_csv(out_csv, rows)
    print(f"[csratio] System B: {len(rows)} sentence rows -> {out_csv}")


def analyse_system_baseline(jsonl_path: pathlib.Path, out_csv: pathlib.Path, system_label: str):
    """Systems A/C: batch-level LLM ratio only — compute deterministic per sentence."""
    rows = []
    for line_no, line in enumerate(_iter_jsonl(jsonl_path)):
        sentences = line.get("data_generation_result", [])
        batch_llm = line.get("cs_ratio_result", {}) or {}
        for i, sentence in enumerate(sentences):
            det = compute_true_cs_stats(sentence)
            rows.append({
                "system": system_label,
                "line": line_no + 1,
                "sentence_index": i,
                "task": line.get("task", ""),
                "topic": line.get("topic", ""),
                "sentence": sentence,
                "det_ar_pct": det.get("ar_pct"),
                "det_en_pct": det.get("en_pct"),
                "det_total_tokens": det.get("total_tokens"),
                "batch_llm_ratio_score": batch_llm.get("ratio_score"),
                "batch_llm_computed_ratio": batch_llm.get("computed_ratio"),
                "configured_cs_ratio": line.get("cs_ratio", ""),
            })
    _write_csv(out_csv, rows)
    print(f"[csratio] {system_label}: {len(rows)} sentence rows -> {out_csv}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_jsonl(path: pathlib.Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_csv(path: pathlib.Path, rows: list[dict]):
    if not rows:
        print(f"[csratio] No rows to write for {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS ratio validation: deterministic vs LLM.")
    parser.add_argument("--systems", nargs="+", choices=["a", "b", "c"], default=["a", "b", "c"])
    args = parser.parse_args()

    CSRATIO_OUT.mkdir(parents=True, exist_ok=True)

    if "b" in args.systems:
        analyse_system_b(SYSTEMS["b"], CSRATIO_OUT / "cs_ratio_validation_B.csv")
    if "a" in args.systems:
        analyse_system_baseline(SYSTEMS["a"], CSRATIO_OUT / "cs_ratio_validation_A.csv", "System_A")
    if "c" in args.systems:
        analyse_system_baseline(SYSTEMS["c"], CSRATIO_OUT / "cs_ratio_validation_C.csv", "System_C")

    print("[csratio] Done.")
