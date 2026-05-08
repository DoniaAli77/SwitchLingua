"""
run_llm_ratio_repeats.py
========================
Thesis Experiment — LLM CS Ratio Consistency

Runs the same N scenarios repeatedly (default 3 repeats) and compares
the LLM-reported CS ratio across runs to measure reliability/variance.

Motivation: The deterministic counter gives one fixed answer; the LLM may
give different answers for the same sentence on different calls. This script
quantifies that variance.

For System B the per-instance LLM ratio from cs_ratio_results_per_instances
is collected across repeats and summary statistics are computed.

Output:
  experiments/outputs/switchlingua/csratio/
    llm_ratio_repeats_B.csv     ← raw per-sentence per-repeat data
    llm_ratio_variance_B.csv    ← std-dev / range summary per sentence

Usage:
    python experiments/switchlingua/run_llm_ratio_repeats.py [--repeats N] [--max-scenarios M]
"""

import argparse
import asyncio
import csv
import json
import pathlib
import sys
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
if str(MODIFIED_CORE) not in sys.path:
    sys.path.insert(0, str(MODIFIED_CORE))

from run_french import CodeSwitchingAgent as ModifiedAgent  # noqa: E402
from utils import load_config, generate_scenarios             # noqa: E402

CONFIG_DEFAULT = ROOT / "Modified_Version" / "config" / "config2.yaml"
OUT_DIR = ROOT / "experiments" / "outputs" / "switchlingua" / "csratio"


async def run_once(scenarios: list) -> list[dict]:
    results = []
    for scenario in scenarios:
        agent = ModifiedAgent(scenario)
        state = await agent.run()
        results.append(state)
    return results


async def main(config_path: pathlib.Path, repeats: int, max_scenarios: int | None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config(str(config_path))
    pre = config.get("pre_execute", config)
    scenarios = generate_scenarios(pre)
    if max_scenarios:
        scenarios = scenarios[:max_scenarios]

    print(f"[llm-repeats] {len(scenarios)} scenarios × {repeats} repeats")

    # key = (scenario_index, sentence_index)
    ratio_scores_by_key: dict[tuple, list[float]] = {}
    raw_rows = []

    for repeat_idx in range(repeats):
        print(f"[llm-repeats] Repeat {repeat_idx + 1}/{repeats} ...")
        run_results = await run_once(scenarios)
        for sc_idx, state in enumerate(run_results):
            sentences = state.get("data_generation_result", [])
            ratios = state.get("cs_ratio_results_per_instances", [])
            for sent_idx, sentence in enumerate(sentences):
                entry = ratios[sent_idx] if sent_idx < len(ratios) else {}
                score = entry.get("ratio_score")
                key = (sc_idx, sent_idx)
                ratio_scores_by_key.setdefault(key, []).append(score)
                raw_rows.append({
                    "repeat": repeat_idx + 1,
                    "scenario_index": sc_idx,
                    "sentence_index": sent_idx,
                    "task": state.get("task", ""),
                    "label": state.get("label", ""),
                    "sentence": sentence,
                    "llm_ratio_score": score,
                    "llm_computed_ratio": entry.get("computed_ratio", ""),
                })

    # Write raw repeats
    raw_path = OUT_DIR / "llm_ratio_repeats_B.csv"
    _write_csv(raw_path, raw_rows)
    print(f"[llm-repeats] Raw repeats -> {raw_path}")

    # Compute variance summary
    variance_rows = []
    for (sc_idx, sent_idx), scores in ratio_scores_by_key.items():
        valid = [s for s in scores if s is not None]
        variance_rows.append({
            "scenario_index": sc_idx,
            "sentence_index": sent_idx,
            "n_repeats": len(scores),
            "n_valid": len(valid),
            "mean_ratio_score": statistics.mean(valid) if valid else None,
            "stdev_ratio_score": statistics.stdev(valid) if len(valid) > 1 else 0.0,
            "min_ratio_score": min(valid) if valid else None,
            "max_ratio_score": max(valid) if valid else None,
            "range_ratio_score": (max(valid) - min(valid)) if valid else None,
        })

    var_path = OUT_DIR / "llm_ratio_variance_B.csv"
    _write_csv(var_path, variance_rows)
    print(f"[llm-repeats] Variance summary -> {var_path}")


def _write_csv(path: pathlib.Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure LLM CS ratio consistency across repeated runs.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeat runs (default: 3)")
    parser.add_argument("--max-scenarios", type=int, default=5,
                        help="Number of scenarios per repeat (default: 5 for cost control)")
    parser.add_argument("--config", type=pathlib.Path, default=CONFIG_DEFAULT)
    args = parser.parse_args()

    asyncio.run(main(args.config, args.repeats, args.max_scenarios))
