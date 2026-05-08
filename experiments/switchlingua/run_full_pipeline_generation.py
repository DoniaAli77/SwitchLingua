"""
run_full_pipeline_generation.py
================================
Thesis Experiment — Full Pipeline Generation

Runs all three systems (A, B, C) over a shared config and writes JSONL outputs
to experiments/outputs/switchlingua/{system_a,system_b,system_c}/.

System A = Original_baseLine  + GPT-4o          (original architecture, original model)
System B = Modified_Version   + GPT-4o-mini     (modified architecture, cost-reduced model)
System C = Original_baseLine  + GPT-4o-mini     (original architecture, cost-reduced model — model control)

Usage:
    python experiments/switchlingua/run_full_pipeline_generation.py [--config PATH] [--systems a b c]
"""

import argparse
import asyncio
import importlib
import json
import os
import pathlib
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINE_CORE = ROOT / "Original_baseLine" / "core"
MODIFIED_CORE = ROOT / "Modified_Version" / "core"

# Modules that exist in both core directories and must be re-loaded when
# switching between systems in the same process.
_SHARED_MODULES = frozenset([
    "utils", "node_engine", "node_models", "prompt",
    "mcp_tools", "agents", "run_french",
])


def _activate_core(core_dir: pathlib.Path):
    """Swap the active core directory on sys.path and evict cached core modules
    so the next bare import resolves from core_dir."""
    for d in [BASELINE_CORE, MODIFIED_CORE]:
        try:
            sys.path.remove(str(d))
        except ValueError:
            pass
    sys.path.insert(0, str(core_dir))
    for name in _SHARED_MODULES:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXPERIMENT_DIR = ROOT / "experiments"
CONFIG_DEFAULT = ROOT / "Modified_Version" / "config" / "config2.yaml"
OUT_ROOT = EXPERIMENT_DIR / "outputs" / "switchlingua"
OUT_A = OUT_ROOT / "system_a"
OUT_B = OUT_ROOT / "system_b"
OUT_C = OUT_ROOT / "system_c"

# ---------------------------------------------------------------------------
# System runners
# Each function receives the agent class and utils functions as parameters
# so that _activate_core in main() controls which core is live.
# ---------------------------------------------------------------------------

async def _run_scenarios(Agent, load_config_fn, generate_scenarios_fn,
                         config_path: pathlib.Path, out_dir: pathlib.Path,
                         label: str) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config_fn(str(config_path))
    pre = config.get("pre_execute", config)
    scenarios = generate_scenarios_fn(pre)
    results = []
    for i, scenario in enumerate(scenarios):
        tag = f"{scenario.get('task', '')}/{scenario.get('label', scenario.get('topic', ''))}"
        print(f"[{label}] {i+1}/{len(scenarios)}: {tag}")
        agent = Agent(scenario)
        final_state = await agent.run()
        results.append(final_state)
        _append_jsonl(out_dir / "Arabic.jsonl", final_state)
    print(f"[{label}] Done. {len(results)} scenarios -> {out_dir / 'Arabic.jsonl'}")
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _append_jsonl(path: pathlib.Path, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# Systems run sequentially — each needs its own core directory active and
# sys.path/sys.modules cannot safely be shared across both cores in parallel.
# ---------------------------------------------------------------------------

async def main(config_path: pathlib.Path, systems: list[str]):
    if "a" in systems:
        _activate_core(BASELINE_CORE)
        import run_french as _rf  # noqa: PLC0415
        import utils as _ut       # noqa: PLC0415
        await _run_scenarios(_rf.CodeSwitchingAgent, _ut.load_config,
                              _ut.generate_scenarios, config_path, OUT_A, "System A")

    if "b" in systems:
        _activate_core(MODIFIED_CORE)
        import run_french as _rf  # noqa: PLC0415
        import utils as _ut       # noqa: PLC0415
        await _run_scenarios(_rf.CodeSwitchingAgent, _ut.load_config,
                              _ut.generate_scenarios, config_path, OUT_B, "System B")

    if "c" in systems:
        # System C: baseline code + gpt-4o-mini (model control)
        _activate_core(BASELINE_CORE)
        import run_french as _rf   # noqa: PLC0415
        import utils as _ut        # noqa: PLC0415
        import node_engine as _ne  # noqa: PLC0415
        _ne.MODEL = "gpt-4o-mini"  # patch model before any agent is instantiated
        await _run_scenarios(_rf.CodeSwitchingAgent, _ut.load_config,
                              _ut.generate_scenarios, config_path, OUT_C, "System C")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full pipeline generation for all thesis systems.")
    parser.add_argument("--config", type=pathlib.Path, default=CONFIG_DEFAULT,
                        help="Path to config YAML (default: Modified_Version/config/config2.yaml)")
    parser.add_argument("--systems", nargs="+", choices=["a", "b", "c"], default=["a", "b", "c"],
                        help="Which systems to run (default: all)")
    args = parser.parse_args()

    asyncio.run(main(args.config, args.systems))
