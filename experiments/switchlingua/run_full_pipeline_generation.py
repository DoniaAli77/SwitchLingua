"""
run_full_pipeline_generation.py
================================
Thesis Experiment — Full Pipeline Generation

Runs one or more systems over a shared config and writes JSONL outputs to:
  experiments/outputs/switchlingua/system_b_modified_mini/
  experiments/outputs/switchlingua/system_c_original_mini/
  experiments/outputs/switchlingua/system_a_original_gpt4o/   (expensive — opt-in only)

System A = Original_baseLine  + GPT-4o          (original architecture, original model)
System B = Modified_Version   + GPT-4o-mini     (modified architecture, cost-reduced model)
System C = Original_baseLine  + GPT-4o-mini     (original architecture, cost-reduced model — model control)

Usage:
    # Run B and C with 5 scenarios each (default):
    python experiments/switchlingua/run_full_pipeline_generation.py

    # Run only B with 10 scenarios:
    python experiments/switchlingua/run_full_pipeline_generation.py --systems b --count 10

    # Include System A (costly — uses GPT-4o):
    python experiments/switchlingua/run_full_pipeline_generation.py --systems a b c --count 5
"""

import argparse
import asyncio
import importlib
import json
import os
import pathlib
import sys

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINE_CORE = ROOT / "Original_baseLine" / "core"
MODIFIED_CORE  = ROOT / "Modified_Version"  / "core"
CONFIG_DEFAULT = ROOT / "Modified_Version" / "config" / "config2.yaml"

# ---------------------------------------------------------------------------
# Load .env — Modified_Version/.env contains OPENAI_API_KEY / OPENAI_BASE_URL.
# Also set API_KEY / API_BASE aliases so Original_baseLine's dotenv.load_dotenv()
# finds the same credentials regardless of which core is active.
# ---------------------------------------------------------------------------
import dotenv as _dotenv

_env_path = ROOT / "Modified_Version" / ".env"
if _env_path.exists():
    _dotenv.load_dotenv(str(_env_path), override=True)  # .env always wins over stale Windows env

# Always sync aliases so both cores see the same credentials.
os.environ["API_KEY"]  = os.environ.get("OPENAI_API_KEY", "")
os.environ["API_BASE"] = os.environ.get("OPENAI_BASE_URL", "")

# ---------------------------------------------------------------------------
# SSL — corporate/university proxies do TLS inspection and present a self-signed
# cert that Python's ssl module rejects.  Patch httpx (used by the OpenAI SDK)
# to skip certificate verification globally before any clients are created.
# ---------------------------------------------------------------------------
import ssl as _ssl
import httpx as _httpx

_ssl._create_default_https_context = _ssl._create_unverified_context  # noqa: SLF001

_httpx_Client_orig = _httpx.Client.__init__
def _httpx_client_no_verify(self, *a, **kw):
    kw.setdefault("verify", False)
    _httpx_Client_orig(self, *a, **kw)
_httpx.Client.__init__ = _httpx_client_no_verify

_httpx_AsyncClient_orig = _httpx.AsyncClient.__init__
def _httpx_async_client_no_verify(self, *a, **kw):
    kw.setdefault("verify", False)
    _httpx_AsyncClient_orig(self, *a, **kw)
_httpx.AsyncClient.__init__ = _httpx_async_client_no_verify

# ---------------------------------------------------------------------------
# Module names shared between both cores — must be evicted on core switch.
# ---------------------------------------------------------------------------
_SHARED_MODULES = frozenset([
    "utils", "node_engine", "node_models", "prompt",
    "mcp_tools", "agents", "run_french",
])


def _activate_core(core_dir: pathlib.Path) -> None:
    """Swap the active core directory on sys.path and evict cached core modules."""
    for d in (BASELINE_CORE, MODIFIED_CORE):
        try:
            sys.path.remove(str(d))
        except ValueError:
            pass
    sys.path.insert(0, str(core_dir))
    for name in _SHARED_MODULES:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUT_ROOT = ROOT / "experiments" / "outputs" / "switchlingua"
OUT_A    = OUT_ROOT / "system_a_original_gpt4o"
OUT_B    = OUT_ROOT / "system_b_modified_mini"
OUT_C    = OUT_ROOT / "system_c_original_mini"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_serializable(obj):
    """Recursively convert non-JSON-serializable types (set, etc.) to JSON-safe equivalents."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_make_serializable(v) for v in obj)
    return obj


def _append_jsonl(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_make_serializable(record), ensure_ascii=False) + "\n")


CS_TYPES_ALL = [
    "Intrasentential",
    "Intersentential",
    "Extra-sentential / Tag switching",
]


async def _run_scenarios(
    Agent,
    scenarios: list,
    out_dir: pathlib.Path,
    label: str,
    count: int,
) -> list:
    scenarios = scenarios[:count]
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    failed = 0
    for i, scenario in enumerate(scenarios):
        task  = scenario.get("task",  "topic")
        topic = scenario.get("label", scenario.get("topic", ""))
        cs    = scenario.get("cs_type", "")
        print(f"[{label}] {i + 1}/{len(scenarios)}: task={task}  topic={topic}  cs_type={cs}")
        try:
            agent = Agent(scenario)
            state = await agent.run() or {}
            results.append(state)
            # AcceptanceAgent already writes to out_dir/Arabic.jsonl directly.
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[{label}] SKIPPED scenario {i + 1} ({task}/{topic}/{cs}): {type(exc).__name__}: {exc}")
    print(f"[{label}] Done — {len(results)} succeeded, {failed} failed → {out_dir / 'Arabic.jsonl'}")
    return results


# ---------------------------------------------------------------------------
# Per-system runners
# Systems run sequentially — each needs its own core active; sys.path/
# sys.modules cannot safely be shared across both cores in parallel.
# ---------------------------------------------------------------------------

async def run_system_b(
    config_path: pathlib.Path,
    count: int,
    cs_types: list[str] | None = None,
    task_filter: str | None = None,
) -> None:
    """System B: Modified_Version + gpt-4o-mini."""
    _activate_core(MODIFIED_CORE)
    import run_french  as _rf  # noqa: PLC0415
    import utils       as _ut  # noqa: PLC0415
    import node_engine as _ne  # noqa: PLC0415

    _ne.MODEL      = "gpt-4o-mini"
    _ne.OUTPUT_DIR = str(OUT_B)

    config = _ut.load_config(str(config_path))
    if cs_types:
        config["pre_execute"]["shared"]["cs_type"] = cs_types
    scenarios = _ut.generate_scenarios(config["pre_execute"])
    if task_filter:
        scenarios = [s for s in scenarios if s.get("task") == task_filter]
    await _run_scenarios(_rf.CodeSwitchingAgent, scenarios, OUT_B, "System B", count)


async def run_system_c(
    count: int,
    cs_types: list[str] | None = None,
    topic: str = "tech",
) -> None:
    """System C: Original_baseLine + gpt-4o-mini (model-control baseline).

    Original_baseLine's generate_scenarios expects a flat config dict with a
    top-level 'topics' list — not the pre_execute nesting used by config2.yaml.
    A programmatic config is built here with enough topics for `count` scenarios.
    """
    _activate_core(BASELINE_CORE)
    import run_french  as _rf  # noqa: PLC0415
    import utils       as _ut  # noqa: PLC0415
    import node_engine as _ne  # noqa: PLC0415

    _ne.MODEL      = "gpt-4o-mini"
    _ne.OUTPUT_DIR = str(OUT_C)

    cfg = {
        "topics":            [topic],
        "tense":             ["Present"],
        "perspective":       ["First Person"],
        "cs_ratio":          ["70%"],
        "character_setting": {
            "nationality":     {"first_language": "Arabic", "second_language": "English"},
            "gender":          ["Male"],
            "age":             ["18-25"],
            "education_level": ["College"],
        },
        "conversation_type": ["single_turn"],
        "cs_function":       ["Expressive"],
        "cs_type":           cs_types if cs_types else ["Intrasentential"],
        "use_tools":         False,
    }
    scenarios = _ut.generate_scenarios(cfg)
    for s in scenarios:
        s.setdefault("mcp_result", "")
    await _run_scenarios(_rf.CodeSwitchingAgent, scenarios, OUT_C, "System C", count)


async def run_system_a(count: int) -> None:
    """System A: Original_baseLine + gpt-4o (expensive — opt-in only)."""
    _activate_core(BASELINE_CORE)
    import run_french  as _rf  # noqa: PLC0415
    import utils       as _ut  # noqa: PLC0415
    import node_engine as _ne  # noqa: PLC0415

    _ne.MODEL      = "gpt-4o"
    _ne.OUTPUT_DIR = str(OUT_A)

    cfg = {
        "topics": [
            "tech", "finance", "business", "education", "health",
            "sports", "travel", "food", "culture", "science",
        ],
        "tense":             ["Present"],
        "perspective":       ["First Person"],
        "cs_ratio":          ["70%"],
        "character_setting": {
            "nationality":     {"first_language": "Arabic", "second_language": "English"},
            "gender":          ["Male"],
            "age":             ["18-25"],
            "education_level": ["College"],
        },
        "conversation_type": ["single_turn"],
        "cs_function":       ["Expressive"],
        "cs_type":           ["Intrasentential"],
        "use_tools":         False,
    }
    scenarios = _ut.generate_scenarios(cfg)
    for s in scenarios:
        s.setdefault("mcp_result", "")
    await _run_scenarios(_rf.CodeSwitchingAgent, scenarios, OUT_A, "System A", count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(
    config_path: pathlib.Path,
    systems: list[str],
    count: int,
    cs_types: list[str] | None = None,
    task_filter: str | None = None,
    topic: str = "tech",
) -> None:
    cs_label = ", ".join(cs_types) if cs_types else "(config default)"
    print(f"Running systems={systems}  count={count}  config={config_path}")
    print(f"  cs_types={cs_label}  task_filter={task_filter or 'all'}  topic={topic}")
    if "b" in systems:
        await run_system_b(config_path, count, cs_types=cs_types, task_filter=task_filter)
    if "c" in systems:
        await run_system_c(count, cs_types=cs_types, topic=topic)
    if "a" in systems:
        await run_system_a(count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate fresh thesis outputs for SwitchLingua systems."
    )
    parser.add_argument(
        "--config", type=pathlib.Path, default=CONFIG_DEFAULT,
        help="Config YAML for System B (default: Modified_Version/config/config2.yaml)",
    )
    parser.add_argument(
        "--systems", nargs="+", choices=["a", "b", "c"], default=["b", "c"],
        help="Systems to run. Default: b c  (System A is expensive — opt-in only)",
    )
    parser.add_argument(
        "--count", type=int, default=5,
        help="Maximum number of scenarios per system (default: 5)",
    )
    parser.add_argument(
        "--cs_types", nargs="+", default=None,
        metavar="CS_TYPE",
        help=(
            "Override cs_type list for both systems. "
            "E.g.: --cs_types Intrasentential Intersentential \"Extra-sentential / Tag switching\""
        ),
    )
    parser.add_argument(
        "--task", default=None,
        help="Filter System B scenarios to this task only (e.g. topic).",
    )
    parser.add_argument(
        "--topic", default="tech",
        help="Topic used when building System C scenarios programmatically (default: tech).",
    )
    args = parser.parse_args()

    asyncio.run(main(
        args.config, args.systems, args.count,
        cs_types=args.cs_types,
        task_filter=args.task,
        topic=args.topic,
    ))
