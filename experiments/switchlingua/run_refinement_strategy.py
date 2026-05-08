"""
run_refinement_strategy.py
===========================
Thesis Experiment — Refinement Strategy Comparison

Compares three refinement strategies:
  per_sentence   → Modified_Version default: refine only failing sentences,
                   then re-evaluate quality agents (fresh scores)
  scenario_all   → Baseline-style: rewrite all sentences on any failure,
                   do NOT re-evaluate quality agents after rewriting
  none           → No refinement at all; accept first-pass output as-is

For a fair comparison, all three strategies run the same scenarios using the
Modified_Version pipeline architecture, but with different refinement knobs.

Output:
  experiments/outputs/switchlingua/refinement/
    per_sentence/Arabic.jsonl
    scenario_all/Arabic.jsonl
    none/Arabic.jsonl
    refinement_strategy_comparison.csv   ← aggregate score comparison

Usage:
    python experiments/switchlingua/run_refinement_strategy.py [--max-scenarios N]
"""

import argparse
import asyncio
import csv
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
if str(MODIFIED_CORE) not in sys.path:
    sys.path.insert(0, str(MODIFIED_CORE))

from run_french import CodeSwitchingAgent as ModifiedAgent  # noqa: E402
from utils import load_config, generate_scenarios             # noqa: E402

CONFIG_DEFAULT = ROOT / "Modified_Version" / "config" / "config2.yaml"
OUT_ROOT = ROOT / "experiments" / "outputs" / "switchlingua" / "refinement"

# Env var overrides for each strategy
STRATEGIES = {
    "per_sentence": {},
    "scenario_all": {"REFINE_STRATEGY": "scenario_all", "ENABLE_REEVAL": "0"},
    "none":         {"MAX_SENTENCE_REFINES": "0"},
}


async def run_strategy(name: str, env_overrides: dict, scenarios: list, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "Arabic.jsonl"
    saved = {k: os.environ.get(k) for k in env_overrides}
    os.environ.update(env_overrides)
    results = []
    try:
        print(f"[refinement:{name}] Running {len(scenarios)} scenarios ...")
        for i, scenario in enumerate(scenarios):
            agent = ModifiedAgent(scenario)
            state = await agent.run()
            state["_strategy"] = name
            results.append(state)
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(state, ensure_ascii=False) + "\n")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print(f"[refinement:{name}] Done -> {out_file}")
    return results


def summarise(all_results: dict[str, list]) -> list[dict]:
    rows = []
    for strategy, records in all_results.items():
        scores = [r.get("score") for r in records if r.get("score") is not None]
        refine_counts = [r.get("refine_count", 0) for r in records]
        rows.append({
            "strategy": strategy,
            "n_scenarios": len(records),
            "avg_score": sum(scores) / len(scores) if scores else None,
            "avg_refine_count": sum(refine_counts) / len(refine_counts) if refine_counts else 0,
            "pct_refined": sum(1 for rc in refine_counts if rc > 0) / len(refine_counts) * 100 if refine_counts else 0,
        })
    return rows


async def main(config_path: pathlib.Path, max_scenarios: int | None):
    config = load_config(str(config_path))
    pre = config.get("pre_execute", config)
    scenarios = generate_scenarios(pre)
    if max_scenarios:
        scenarios = scenarios[:max_scenarios]

    all_results: dict[str, list] = {}
    for name, overrides in STRATEGIES.items():
        all_results[name] = await run_strategy(name, overrides, scenarios, OUT_ROOT / name)

    # Write comparison summary
    summary_rows = summarise(all_results)
    summary_path = OUT_ROOT / "refinement_strategy_comparison.csv"
    if summary_rows:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"\n[refinement] Summary -> {summary_path}")
    for row in summary_rows:
        print(f"  {row['strategy']:20s}  avg_score={row['avg_score']:.3f}  avg_refine={row['avg_refine_count']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare refinement strategies.")
    parser.add_argument("--config", type=pathlib.Path, default=CONFIG_DEFAULT)
    parser.add_argument("--max-scenarios", type=int, default=None)
    args = parser.parse_args()

    asyncio.run(main(args.config, args.max_scenarios))
