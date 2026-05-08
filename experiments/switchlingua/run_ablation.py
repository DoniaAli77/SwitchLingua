"""
run_ablation.py
===============
Thesis Experiment — Ablation Study

Runs the Modified_Version pipeline with individual components disabled one at a
time to measure each component's contribution to output quality.

Ablation conditions (controlled by environment variables read inside the core):
  full          → System B baseline (all components on)
  no_tv         → TaskValidatorAgent disabled (ENABLE_TASK_VALIDATOR=0)
  no_reeval     → Post-refinement re-evaluation disabled (ENABLE_REEVAL=0)
  no_det_cs     → Deterministic CS stats not injected; LLM-only CS ratio (DET_CS_STATS=0)
  no_refine     → Refinement disabled; accept all first-pass sentences (MAX_SENTENCE_REFINES=0)
  no_per_sent   → Per-sentence scoring disabled; fall back to scenario-level score (PER_SENT_SCORING=0)

Each condition writes output to:
  experiments/outputs/switchlingua/ablation/<condition>/Arabic.jsonl

Usage:
    python experiments/switchlingua/run_ablation.py [--conditions full no_tv ...] [--max-scenarios N]
"""

import argparse
import asyncio
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

# ---------------------------------------------------------------------------
# Ablation condition definitions
# Each entry is a dict of env-var overrides to apply before running.
# ---------------------------------------------------------------------------
ABLATION_CONDITIONS: dict[str, dict[str, str]] = {
    "full": {},
    "no_tv":       {"ENABLE_TASK_VALIDATOR": "0"},
    "no_reeval":   {"ENABLE_REEVAL": "0"},
    "no_det_cs":   {"DET_CS_STATS": "0"},
    "no_refine":   {"MAX_SENTENCE_REFINES": "0"},
    "no_per_sent": {"PER_SENT_SCORING": "0"},
}

CONFIG_DEFAULT = ROOT / "Modified_Version" / "config" / "config2.yaml"
OUT_ROOT = ROOT / "experiments" / "outputs" / "switchlingua" / "ablation"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _apply_env(overrides: dict[str, str]):
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    return saved


def _restore_env(saved: dict[str, str | None]):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


async def run_condition(condition_name: str, overrides: dict[str, str],
                        scenarios: list, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "Arabic.jsonl"
    saved = _apply_env(overrides)
    print(f"[ablation:{condition_name}] Running {len(scenarios)} scenarios ...")
    results = []
    try:
        for i, scenario in enumerate(scenarios):
            agent = ModifiedAgent(scenario)
            state = await agent.run()
            state["_ablation_condition"] = condition_name
            results.append(state)
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(state, ensure_ascii=False) + "\n")
    finally:
        _restore_env(saved)
    print(f"[ablation:{condition_name}] Done -> {out_file}")
    return results


async def main(config_path: pathlib.Path, conditions: list[str], max_scenarios: int | None):
    config = load_config(str(config_path))
    pre = config.get("pre_execute", config)
    scenarios = generate_scenarios(pre)
    if max_scenarios:
        scenarios = scenarios[:max_scenarios]

    for cond in conditions:
        overrides = ABLATION_CONDITIONS[cond]
        out_dir = OUT_ROOT / cond
        await run_condition(cond, overrides, scenarios, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study on Modified_Version pipeline.")
    parser.add_argument("--conditions", nargs="+",
                        choices=list(ABLATION_CONDITIONS.keys()),
                        default=list(ABLATION_CONDITIONS.keys()),
                        help="Which ablation conditions to run (default: all)")
    parser.add_argument("--config", type=pathlib.Path, default=CONFIG_DEFAULT)
    parser.add_argument("--max-scenarios", type=int, default=None)
    args = parser.parse_args()

    asyncio.run(main(args.config, args.conditions, args.max_scenarios))
