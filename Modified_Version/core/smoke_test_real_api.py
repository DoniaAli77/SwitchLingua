"""
Smoke test: runs 2 real scenarios against the live API using the current
production graph (agents.py / node_engine.py).

Usage:
    python smoke_test_real_api.py

Output is written to output/{first_language}.jsonl as usual.
"""

import langchain
import asyncio
import json
import random
import os
import argparse

# --- Compatibility patch for older LangChain assumptions ---
if not hasattr(langchain, "debug"):
    langchain.debug = False
if not hasattr(langchain, "verbose"):
    langchain.verbose = False
if not hasattr(langchain, "llm_cache"):
    langchain.llm_cache = None

import sys
from pathlib import Path

# Ensure imports resolve correctly regardless of working directory
_CORE_DIR = Path(__file__).resolve().parent
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

_DEFAULT_CONFIG_PATH = str(_CORE_DIR.parent / "config" / "config2.yaml")

from loguru import logger
from datetime import datetime
from utils import load_config, generate_scenarios
from agents import CodeSwitchingAgent

logger.add(
    f"logs/smoke_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level="INFO",
)

MAX_SMOKE_SCENARIOS = 2


def _to_console_safe(text: str) -> str:
    """Return text safely encodable by current stdout (Windows cp1252-safe)."""
    if text is None:
        return ""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(text).encode(enc, errors="replace").decode(enc, errors="replace")


def resolve_config_path(cli_config: str | None) -> str:
    """Resolve config path from CLI arg, env var, or default path."""
    if cli_config:
        return cli_config
    env_config = os.getenv("SWITCHLINGUA_CONFIG_PATH")
    if env_config:
        return env_config
    return _DEFAULT_CONFIG_PATH


async def run_one(scenario: dict, index: int) -> dict | None:
    try:
        agent = CodeSwitchingAgent(scenario)
        result = await agent.run()
        sentences = result.get("data_generation_result", []) if result else []
        scores = result.get("sentence_scores", []) if result else []
        records = result.get("sentence_records", []) if result else []
        refine_counts = result.get("instance_refine_counts", []) if result else []

        print(f"\n{'='*60}")
        print(f"Scenario #{index}  task={scenario.get('task')}  topic={scenario.get('topic')}")
        print(f"  Sentences generated : {len(sentences)}")
        for i, (s, sc) in enumerate(zip(sentences, scores)):
            rc = refine_counts[i] if i < len(refine_counts) else 0
            rec = records[i] if i < len(records) else {}
            task_passed = rec.get("task_validation", {}).get("passed", "?")
            print(f"  [{i}] score={sc:.2f}  refined={rc}x  task_passed={task_passed}")
            print(f"       {_to_console_safe(s[:120])}")
        print(f"  Overall score       : {result.get('score', '?')}")
        print(f"{'='*60}")
        return result
    except Exception as e:
        logger.exception(f"Scenario #{index} FAILED: {e}")
        print(f"\n[FAILED] Scenario #{index}: {type(e).__name__}: {e}")
        return None


async def main(config_path: str, max_scenarios: int) -> bool:
    config = load_config(config_path)
    scenarios = generate_scenarios(config["pre_execute"])
    random.shuffle(scenarios)
    smoke = scenarios[:max_scenarios]

    print(f"Running {len(smoke)} smoke scenario(s) against live API...")
    results = await asyncio.gather(*[run_one(s, i) for i, s in enumerate(smoke)])

    passed = sum(1 for r in results if r is not None)
    print(f"\nSmoke test done: {passed}/{len(smoke)} scenarios completed successfully.")
    if passed < len(smoke):
        print("Some scenarios FAILED — check logs/ for details.")
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run smoke scenarios with real API.")
    parser.add_argument("--config", default=None, help="Path to config yaml.")
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=MAX_SMOKE_SCENARIOS,
        help="Number of scenarios to run.",
    )
    args = parser.parse_args()

    cfg_path = resolve_config_path(args.config)
    print(f"Using config: {cfg_path}")
    ok = asyncio.run(main(cfg_path, args.max_scenarios))
    sys.exit(0 if ok else 1)
