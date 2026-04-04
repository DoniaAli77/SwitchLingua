import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


def _resolve_paths() -> tuple[Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    core_dir = project_root / "core"
    default_config = project_root / "config" / "config2.yaml"
    default_env = project_root / ".env"
    return core_dir, default_config, default_env


def _load_dependencies(core_dir: Path):
    sys.path.insert(0, str(core_dir))

    from utils import load_config, generate_scenarios, weighting_scheme  # type: ignore
    from node_engine import (  # type: ignore
        RunDataGenerationAgent,
        RunTaskValidatorAgent,
        RunFluencyAgent,
        RunNaturalnessAgent,
        RunCSRatioAgent,
        RunSocialCulturalAgent,
        SummarizeResult,
    )

    return (
        load_config,
        generate_scenarios,
        weighting_scheme,
        RunDataGenerationAgent,
        RunTaskValidatorAgent,
        RunFluencyAgent,
        RunNaturalnessAgent,
        RunCSRatioAgent,
        RunSocialCulturalAgent,
        SummarizeResult,
    )


def _validate_env() -> tuple[bool, str]:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    base = os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE")
    if not key:
        return False, "Missing API key. Set OPENAI_API_KEY or API_KEY."
    if not base:
        return False, "Missing API base URL. Set OPENAI_BASE_URL or API_BASE."
    return True, ""


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(list(value))
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    return value


def _pick_scenarios_per_task(scenarios: list[dict], max_per_task: int) -> dict[str, list[dict]]:
    picked: dict[str, list[dict]] = {"topic": [], "sentiment": [], "ner": []}
    seen_keys: dict[str, set[str]] = {"topic": set(), "sentiment": set(), "ner": set()}

    for scenario in scenarios:
        task = scenario.get("task")
        if task not in picked:
            continue
        if len(picked[task]) >= max_per_task:
            continue

        key = str(scenario.get("label") or scenario.get("topic") or "")
        if key in seen_keys[task]:
            continue

        picked[task].append(scenario)
        seen_keys[task].add(key)

    return picked


def _prepare_state(scenario: dict) -> dict:
    state = dict(scenario)
    state["news_article"] = ""
    state["news_dict"] = {}
    state["news_hash"] = set()
    state["mcp_result"] = ""
    state["refine_count"] = int(state.get("refine_count", 0))
    return state


def _run_one_pipeline(
    state: dict,
    weighting_scheme,
    RunDataGenerationAgent,
    RunTaskValidatorAgent,
    RunFluencyAgent,
    RunNaturalnessAgent,
    RunCSRatioAgent,
    RunSocialCulturalAgent,
    SummarizeResult,
):
    out = {}

    out.update(RunDataGenerationAgent(state))
    state.update(out)

    out.update(RunTaskValidatorAgent(state))
    state.update(out)

    out.update(RunFluencyAgent(state))
    state.update(out)

    out.update(RunNaturalnessAgent(state))
    state.update(out)

    out.update(RunCSRatioAgent(state))
    state.update(out)

    out.update(RunSocialCulturalAgent(state))
    state.update(out)

    summary_out = SummarizeResult(state)
    state.update(summary_out)

    n = len(state.get("data_generation_result", []))
    checks = {
        "instances_count": n,
        "task_validator_exists": isinstance(state.get("task_validation_result"), dict),
        "fluency_per_instances_ok": len(state.get("fluency_results_per_instances", [])) == n,
        "naturalness_per_instances_ok": len(state.get("naturalness_results_per_instances", [])) == n,
        "csratio_per_instances_ok": len(state.get("cs_ratio_results_per_instances", [])) == n,
        "social_per_instances_ok": len(state.get("social_cultural_results_per_instances", [])) == n,
        "has_aggregate_fluency": "fluency_score" in state.get("fluency_result", {}),
        "has_aggregate_naturalness": "naturalness_score" in state.get("naturalness_result", {}),
        "has_aggregate_social": "socio_cultural_score" in state.get("social_cultural_result", {}),
        "score_matches_weighting": abs(float(state.get("score", 0.0)) - float(weighting_scheme(state))) < 1e-9,
    }

    checks["all_passed"] = all(bool(v) for v in checks.values() if isinstance(v, (bool, int, float)))
    return checks


def main() -> None:
    core_dir, default_config, default_env = _resolve_paths()

    parser = argparse.ArgumentParser(description="Run FULL real pipeline test (no mocks).")
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--env-file", default=str(default_env))
    parser.add_argument("--tasks", nargs="+", default=["topic", "sentiment", "ner"])
    parser.add_argument("--max-scenarios-per-task", type=int, default=1)
    args = parser.parse_args()

    env_file = Path(args.env_file)
    if env_file.exists():
        load_dotenv(env_file, override=True)

    ok, msg = _validate_env()
    if not ok:
        print(f"ERROR: {msg}")
        raise SystemExit(1)

    (
        load_config,
        generate_scenarios,
        weighting_scheme,
        RunDataGenerationAgent,
        RunTaskValidatorAgent,
        RunFluencyAgent,
        RunNaturalnessAgent,
        RunCSRatioAgent,
        RunSocialCulturalAgent,
        SummarizeResult,
    ) = _load_dependencies(core_dir)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        raise SystemExit(1)

    config = load_config(str(config_path))
    scenarios = generate_scenarios(config["pre_execute"])

    requested_tasks = [t for t in args.tasks if t in {"topic", "sentiment", "ner"}]
    if not requested_tasks:
        print("ERROR: No valid tasks provided. Use topic/sentiment/ner.")
        raise SystemExit(1)

    picked = _pick_scenarios_per_task(scenarios, max(1, args.max_scenarios_per_task))

    results = {"meta": {}, "runs": []}
    results["meta"] = {
        "config": str(config_path),
        "env_file": str(env_file),
        "tasks": requested_tasks,
        "max_scenarios_per_task": args.max_scenarios_per_task,
        "timestamp": datetime.now().isoformat(),
    }

    print("Running FULL real pipeline test")
    print("Tasks:", requested_tasks)

    for task in requested_tasks:
        selected = picked.get(task, [])
        if not selected:
            print(f"WARN: No scenario found for task={task}")
            continue

        for idx, scenario in enumerate(selected, start=1):
            state = _prepare_state(scenario)
            print(f"\n=== TASK={task} SCENARIO={idx} ===")
            print("Scenario topic:", state.get("topic"), "label:", state.get("label"))

            try:
                checks = _run_one_pipeline(
                    state,
                    weighting_scheme,
                    RunDataGenerationAgent,
                    RunTaskValidatorAgent,
                    RunFluencyAgent,
                    RunNaturalnessAgent,
                    RunCSRatioAgent,
                    RunSocialCulturalAgent,
                    SummarizeResult,
                )

                print(json.dumps(checks, ensure_ascii=False, indent=2))
                print("Final score:", state.get("score"))

                results["runs"].append(
                    {
                        "task": task,
                        "scenario_index": idx,
                        "checks": checks,
                        "state": _to_jsonable(state),
                    }
                )
            except Exception as exc:
                print("ERROR:", str(exc))
                results["runs"].append(
                    {
                        "task": task,
                        "scenario_index": idx,
                        "error": str(exc),
                        "state": _to_jsonable(state),
                    }
                )

    all_passed = True
    for run in results["runs"]:
        if run.get("error"):
            all_passed = False
        elif not run.get("checks", {}).get("all_passed", False):
            all_passed = False

    print("\n=== OVERALL ===")
    print("ALL_PASSED:", all_passed)

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pipeline_full_real_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results), f, ensure_ascii=False, indent=2)

    print("Saved output to:", str(output_path))

    if not all_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
