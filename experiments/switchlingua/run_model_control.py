"""
run_model_control.py
====================
Thesis Experiment — Model Control (System C)

Runs the Original_baseLine pipeline code with GPT-4o-mini as the model,
isolating the effect of model choice from architecture changes.

System C = Original_baseLine code + GPT-4o-mini

Comparing results:
  System A (Original + GPT-4o)       vs System C  →  model effect alone
  System C (Original + GPT-4o-mini)  vs System B  →  architecture effect alone

Usage:
    python experiments/switchlingua/run_model_control.py [--config PATH] [--max-scenarios N]
"""

import argparse
import asyncio
import json
import os
import pathlib
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINE_CORE = ROOT / "Original_baseLine" / "core"
if str(BASELINE_CORE) not in sys.path:
    sys.path.insert(0, str(BASELINE_CORE))

from run_french import CodeSwitchingAgent as BaselineAgent    # noqa: E402
from utils import load_config as baseline_load_config          # noqa: E402
from utils import generate_scenarios as baseline_generate_scenarios  # noqa: E402
import node_engine as _ne  # noqa: E402
_ne.MODEL = "gpt-4o-mini"  # System C: same baseline code, cost-reduced model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CONFIG_DEFAULT = ROOT / "Modified_Version" / "config" / "config2.yaml"
OUT_DIR = ROOT / "experiments" / "outputs" / "switchlingua" / "system_c"

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run(config_path: pathlib.Path, max_scenarios: int | None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "Arabic.jsonl"

    # Override the model so baseline code uses GPT-4o-mini
    os.environ["OPENAI_MODEL_OVERRIDE"] = "gpt-4o-mini"
    print("[System C] OPENAI_MODEL_OVERRIDE set to gpt-4o-mini")

    config = baseline_load_config(str(config_path))
    scenarios = baseline_generate_scenarios(config.get("pre_execute", config))

    if max_scenarios is not None:
        scenarios = scenarios[:max_scenarios]
        print(f"[System C] Capped at {max_scenarios} scenarios")

    print(f"[System C] Running {len(scenarios)} scenarios ...")
    results = []

    for i, scenario in enumerate(scenarios):
        label = f"{scenario.get('task', 'unknown')}/{scenario.get('topic', scenario.get('label', ''))}"
        print(f"[System C] {i+1}/{len(scenarios)} — {label}")
        agent = BaselineAgent(scenario)
        final_state = await agent.run()
        results.append(final_state)
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(final_state, ensure_ascii=False) + "\n")

    print(f"[System C] Complete. {len(results)} records -> {out_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model control experiment: Original code + GPT-4o-mini (System C).")
    parser.add_argument("--config", type=pathlib.Path, default=CONFIG_DEFAULT)
    parser.add_argument("--max-scenarios", type=int, default=None,
                        help="Limit number of scenarios for smoke testing")
    args = parser.parse_args()

    asyncio.run(run(args.config, args.max_scenarios))
