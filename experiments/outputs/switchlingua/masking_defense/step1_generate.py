"""
step1_generate.py — Make examples for the masking defense.
==========================================================
Runs System B (Modified_Version, gpt-4o-mini) and saves the sentences.

Two modes:
  --mode raw    : refiner OFF  ("before" photo) — grade once, no fixing.
  --mode fixed  : refiner ON   ("after"  photo) — the full real system.

All 3 code-switch types are generated. Output goes to a clean folder
(the Arabic.jsonl is wiped at the start of each run so re-runs don't pile up).

Usage:
  python step1_generate.py --mode raw   --count 1          # smoke test
  python step1_generate.py --mode raw   --count 999 --out step1_raw_data
  python step1_generate.py --mode fixed --count 999 --out step1_fixed_data
"""
import argparse
import asyncio
import importlib
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
CONFIG = ROOT / "Modified_Version" / "config" / "config2.yaml"
HERE = pathlib.Path(__file__).resolve().parent

# --- env: load .env, win over stale Windows vars (same as master runner) ---
import dotenv as _dotenv  # noqa: E402
_env = ROOT / "Modified_Version" / ".env"
if _env.exists():
    _dotenv.load_dotenv(str(_env), override=True)
os.environ["API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
os.environ["API_BASE"] = os.environ.get("OPENAI_BASE_URL", "")

# --- SSL: skip verification (corporate TLS proxy) ---
import ssl as _ssl  # noqa: E402
import httpx as _httpx  # noqa: E402
_ssl._create_default_https_context = _ssl._create_unverified_context
# verify=False (TLS proxy) + a timeout so a hung call can never freeze the run.
_HTTP_TIMEOUT = 60.0
_co = _httpx.Client.__init__
def _c(self, *a, **k): k.setdefault("verify", False); k.setdefault("timeout", _HTTP_TIMEOUT); _co(self, *a, **k)
_httpx.Client.__init__ = _c
_ao = _httpx.AsyncClient.__init__
def _ac(self, *a, **k): k.setdefault("verify", False); k.setdefault("timeout", _HTTP_TIMEOUT); _ao(self, *a, **k)
_httpx.AsyncClient.__init__ = _ac

CS_TYPES_ALL = [
    "Intrasentential",
    "Intersentential",
    "Extra-sentential / Tag switching",
]


def _activate_modified_core():
    if str(MODIFIED_CORE) not in sys.path:
        sys.path.insert(0, str(MODIFIED_CORE))
    for name in ("utils", "node_engine", "node_models", "prompt", "mcp_tools", "agents", "run_french"):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()


async def run(mode: str, count: int, out: str, task: str | None = None):
    _activate_modified_core()
    import run_french as rf
    import utils as ut
    import node_engine as ne

    # Point the model + output folder
    ne.MODEL = "gpt-4o-mini"
    out_dir = HERE / out
    out_dir.mkdir(parents=True, exist_ok=True)
    ne.OUTPUT_DIR = str(out_dir)

    # Wipe any previous Arabic.jsonl so this run starts clean
    target = out_dir / "Arabic.jsonl"
    if target.exists():
        target.unlink()

    # --- the two switches that define the mode ---
    if mode == "raw":
        rf.MAX_SENTENCE_REFINES = 0      # refiner OFF: never fixes, grades once
        rf.ENABLE_TASK_VALIDATOR = False  # not needed for raw grades; saves cost
        ne.MAX_SENTENCE_REFINES = 0
    else:  # fixed
        rf.MAX_SENTENCE_REFINES = 1      # refiner ON
        rf.ENABLE_TASK_VALIDATOR = False  # validator OFF: only the refiner differs vs raw
        ne.MAX_SENTENCE_REFINES = 1

    # Build scenarios across ALL 3 code-switch types
    cfg = ut.load_config(str(CONFIG))
    cfg["pre_execute"]["shared"]["cs_type"] = CS_TYPES_ALL
    scenarios = ut.generate_scenarios(cfg["pre_execute"])
    if task:
        scenarios = [s for s in scenarios if s.get("task") == task]
    scenarios = scenarios[:count]

    print(f"[step1] mode={mode}  refiner={'OFF' if mode=='raw' else 'ON'}  "
          f"task_validator={'ON' if rf.ENABLE_TASK_VALIDATOR else 'OFF'}")
    print(f"[step1] scenarios to run: {len(scenarios)}  ->  {target}")

    ok = 0
    failed = 0
    for i, sc in enumerate(scenarios):
        t = sc.get("task", "topic")
        cs = sc.get("cs_type", "")
        print(f"[step1] {i+1}/{len(scenarios)}  task={t}  cs_type={cs}")
        try:
            agent = rf.CodeSwitchingAgent(sc)
            await agent.run()
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"[step1] SKIPPED {i+1}: {type(exc).__name__}: {exc}")

    print(f"[step1] DONE — {ok} ok, {failed} failed -> {target}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["raw", "fixed"], required=True)
    p.add_argument("--count", type=int, default=999)
    p.add_argument("--out", default=None,
                   help="output subfolder under masking_defense/ "
                        "(default: step1_raw_data or step1_fixed_data)")
    p.add_argument("--task", default=None, choices=["topic", "sentiment", "ner"],
                   help="only generate scenarios for this task")
    a = p.parse_args()
    out = a.out or ("step1_raw_data" if a.mode == "raw" else "step1_fixed_data")
    asyncio.run(run(a.mode, a.count, out, task=a.task))
