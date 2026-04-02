import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


def _resolve_paths() -> tuple[Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    core_dir = script_dir / "core"
    default_config = script_dir / "config" / "config2.yaml"
    default_env = core_dir / ".env"
    return core_dir, default_config, default_env


def _load_dependencies(core_dir: Path):
    sys.path.insert(0, str(core_dir))
    from utils import load_config, generate_scenarios  # type: ignore
    from node_engine import RunDataGenerationAgent, RunTaskValidatorAgent  # type: ignore
    from prompt import (  # type: ignore
        DATA_GENERATION_TOPIC_PROMPT,
        DATA_GENERATION_SENTIMENT_PROMPT,
        DATA_GENERATION_NER_PROMPT,
    )

    return (
        load_config,
        generate_scenarios,
        RunDataGenerationAgent,
        RunTaskValidatorAgent,
        DATA_GENERATION_TOPIC_PROMPT,
        DATA_GENERATION_SENTIMENT_PROMPT,
        DATA_GENERATION_NER_PROMPT,
    )


def _scenario_diversity_key(scenario: dict) -> str:
    task = scenario.get("task")
    if task in {"topic", "sentiment"}:
        return str(scenario.get("label") or "")
    if task == "ner":
        constraints = scenario.get("task_constraints", {})
        if isinstance(constraints, dict):
            entity_types = constraints.get("entity_types", [])
            return "|".join([str(scenario.get("topic", ""))] + [str(entity_type) for entity_type in entity_types])
    return str(scenario.get("topic") or "")


def _pick_scenarios_per_task(scenarios: list[dict], max_per_task: int) -> dict[str, list[dict]]:
    picked: dict[str, list[dict]] = {"topic": [], "sentiment": [], "ner": []}
    seen_keys: dict[str, set[str]] = {"topic": set(), "sentiment": set(), "ner": set()}

    for scenario in scenarios:
        task = scenario.get("task")
        if task not in picked:
            continue
        if len(picked[task]) >= max_per_task:
            continue

        key = _scenario_diversity_key(scenario)
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
    return state


def _render_prompt(task: str, state: dict, topic_prompt, sentiment_prompt, ner_prompt) -> str:
    if task == "sentiment":
        prompt_template = sentiment_prompt
    elif task == "ner":
        prompt_template = ner_prompt
    else:
        prompt_template = topic_prompt

    messages = prompt_template.format_messages(**state)
    rendered_parts = []
    for idx, msg in enumerate(messages, start=1):
        rendered_parts.append(f"[Message {idx} | role={msg.type}]\n{msg.content}")
    return "\n\n".join(rendered_parts)


def _validate_env() -> tuple[bool, str]:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    base = os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE")

    if not key:
        return False, "Missing API key. Set OPENAI_API_KEY or API_KEY in environment/.env."
    if not base:
        return False, "Missing API base URL. Set OPENAI_BASE_URL or API_BASE in environment/.env."
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


def _has_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


def _has_english(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def _validate_topic(instances: list[str]) -> dict:
    if not instances:
        return {"passed": False, "reason": "No generated instances."}

    cs_count = sum(1 for item in instances if _has_arabic(item) and _has_english(item))
    passed = cs_count >= 1
    return {
        "passed": passed,
        "reason": "OK" if passed else "No clear Arabic+English code-switching found.",
        "num_instances": len(instances),
        "code_switched_instances": cs_count,
    }


def _validate_sentiment(instances: list[str], label: str) -> dict:
    if not instances:
        return {"passed": False, "reason": "No generated instances.", "label": label}

    pos_words = {
        "love", "great", "good", "amazing", "excited", "helpful", "رائع", "ممتاز", "أحب", "مفيد", "جميل", "متحمس",
    }
    neg_words = {
        "bad", "terrible", "awful", "hate", "frustrating", "worse", "سيء", "سيئة", "أكره", "محبط", "مزعج", "أسوأ",
    }

    joined = " ".join(instances).lower()
    pos_hits = sum(1 for word in pos_words if word in joined)
    neg_hits = sum(1 for word in neg_words if word in joined)

    if label == "positive":
        passed = pos_hits >= 1 and pos_hits >= neg_hits
    elif label == "negative":
        passed = neg_hits >= 1 and neg_hits >= pos_hits
    else:
        passed = abs(pos_hits - neg_hits) <= 1

    return {
        "passed": passed,
        "reason": "OK" if passed else "Lexical polarity check did not match expected label.",
        "label": label,
        "positive_hits": pos_hits,
        "negative_hits": neg_hits,
        "num_instances": len(instances),
    }


def _extract_heuristic_entities(text: str) -> dict[str, int]:
    org_terms = {
        "google", "microsoft", "apple", "meta", "amazon", "openai", "tesla", "samsung", "huawei", "techcrunch",
    }
    loc_terms = {
        "cairo", "dubai", "riyadh", "london", "paris", "new york", "tokyo", "berlin", "beijing", "doha",
    }

    counts = {"PER": 0, "ORG": 0, "LOC": 0}
    text_l = text.lower()

    for org in org_terms:
        if re.search(rf"\b{re.escape(org)}\b", text_l):
            counts["ORG"] += 1

    for loc in loc_terms:
        if loc in text_l:
            counts["LOC"] += 1

    person_like = re.findall(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", text)
    counts["PER"] += len(person_like)
    return counts


def _validate_ner(instances: list[str], task_constraints: dict) -> dict:
    if not instances:
        return {"passed": False, "reason": "No generated instances."}

    combined = " ".join(instances)
    counts = _extract_heuristic_entities(combined)

    min_entities = int(task_constraints.get("min_entities", 0))
    max_entities = int(task_constraints.get("max_entities", 10**9))
    must_types = task_constraints.get("must_include_types", [])

    total_entities = counts["PER"] + counts["ORG"] + counts["LOC"]
    has_required_types = all(counts.get(entity_type, 0) > 0 for entity_type in must_types)
    within_range = min_entities <= total_entities <= max_entities
    passed = has_required_types and within_range

    return {
        "passed": passed,
        "reason": "OK" if passed else "Entity heuristic check failed for count/type constraints.",
        "entity_counts": counts,
        "total_entities": total_entities,
        "min_entities": min_entities,
        "max_entities": max_entities,
        "must_include_types": must_types,
    }


def _validate_task_output(task: str, state: dict, result: dict) -> dict:
    instances = result.get("data_generation_result", []) if isinstance(result, dict) else []

    if task == "sentiment":
        return _validate_sentiment(instances, str(state.get("label", "neutral")))
    if task == "ner":
        constraints = state.get("task_constraints", {}) if isinstance(state.get("task_constraints", {}), dict) else {}
        return _validate_ner(instances, constraints)
    return _validate_topic(instances)


def main():
    core_dir, default_config, default_env = _resolve_paths()

    parser = argparse.ArgumentParser(
        description="Run real data generation for topic/sentiment/ner (no mocks)."
    )
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Path to config YAML (default: drive_code/config/config2.yaml)",
    )
    parser.add_argument(
        "--env-file",
        default=str(default_env),
        help="Path to .env file (default: drive_code/core/.env)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["topic", "sentiment", "ner"],
        help="Tasks to run (subset of: topic sentiment ner)",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print fully rendered prompt messages before calling OpenAI.",
    )
    parser.add_argument(
        "--max-scenarios-per-task",
        type=int,
        default=3,
        help="How many diverse scenarios to run per task (default: 3).",
    )
    args = parser.parse_args()

    env_file = Path(args.env_file)
    if env_file.exists():
        load_dotenv(env_file, override=True)

    ok, error_message = _validate_env()
    if not ok:
        print(f"ERROR: {error_message}")
        raise SystemExit(1)

    (
        load_config,
        generate_scenarios,
        RunDataGenerationAgent,
        RunTaskValidatorAgent,
        topic_prompt,
        sentiment_prompt,
        ner_prompt,
    ) = _load_dependencies(core_dir)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        raise SystemExit(1)

    config = load_config(str(config_path))
    scenarios = generate_scenarios(config["pre_execute"])
    picked = _pick_scenarios_per_task(scenarios, max(1, args.max_scenarios_per_task))

    requested_tasks = [task for task in args.tasks if task in {"topic", "sentiment", "ner"}]
    if not requested_tasks:
        print("ERROR: No valid tasks provided. Use any of: topic sentiment ner")
        raise SystemExit(1)

    missing = [task for task in requested_tasks if not picked.get(task)]
    if missing:
        print(f"ERROR: Could not find scenarios for tasks: {missing}")
        raise SystemExit(1)

    print("Running real generation for tasks:", requested_tasks)
    print("Config:", str(config_path))
    print("Env file:", str(env_file))
    enable_task_validator = os.getenv("ENABLE_TASK_VALIDATOR", "1").strip() == "1"
    print("Task validator enabled:", enable_task_validator)

    outputs: dict[str, list[dict]] = {task: [] for task in requested_tasks}
    states_after_generation: dict[str, list[dict]] = {task: [] for task in requested_tasks}
    validations: dict[str, list[dict]] = {task: [] for task in requested_tasks}

    for task in requested_tasks:
        for scenario_index, scenario in enumerate(picked.get(task, []), start=1):
            state = _prepare_state(scenario)
            try:
                if args.print_prompt:
                    print(f"\n=== {task.upper()} SCENARIO {scenario_index} PROMPT ===")
                    print(_render_prompt(task, state, topic_prompt, sentiment_prompt, ner_prompt))

                result = RunDataGenerationAgent(state)
                state_after = dict(state)
                state_after.update(result)

                validator_result = RunTaskValidatorAgent(state_after) if enable_task_validator else {}
                state_after.update(validator_result)

                output_item = {
                    "scenario_index": scenario_index,
                    "scenario_key": _scenario_diversity_key(scenario),
                    "scenario_summary": {
                        "task": task,
                        "topic": scenario.get("topic"),
                        "label": scenario.get("label"),
                    },
                    **result,
                    **validator_result,
                }

                outputs[task].append(output_item)
                states_after_generation[task].append(_to_jsonable(state_after))
                validations[task].append(_validate_task_output(task, state, result))

                print(f"\n=== {task.upper()} SCENARIO {scenario_index} ===")
                print(json.dumps(output_item, ensure_ascii=False, indent=2))
                if enable_task_validator:
                    print(f"=== {task.upper()} SCENARIO {scenario_index} GLOBAL STATE AFTER TASK VALIDATOR ===")
                else:
                    print(f"=== {task.upper()} SCENARIO {scenario_index} GLOBAL STATE AFTER DATAGENERATION (VALIDATOR DISABLED) ===")
                print(json.dumps(_to_jsonable(state_after), ensure_ascii=False, indent=2))
                print(f"=== {task.upper()} SCENARIO {scenario_index} VALIDATION ===")
                print(json.dumps(validations[task][-1], ensure_ascii=False, indent=2))
            except Exception as exc:
                outputs[task].append({"scenario_index": scenario_index, "error": str(exc)})
                states_after_generation[task].append(_to_jsonable(dict(state)))
                validations[task].append({"passed": False, "reason": str(exc)})
                print(f"\n=== {task.upper()} SCENARIO {scenario_index} ===")
                print("ERROR:", str(exc))

    print("\n=== SUMMARY ===")
    summary = {
        "outputs": outputs,
        "task_validator_enabled": enable_task_validator,
        "states_after_task_validator": states_after_generation,
        "states_after_generation": states_after_generation,
        "validations": validations,
    }
    summary_jsonable = _to_jsonable(summary)
    print(json.dumps(summary_jsonable, ensure_ascii=False, indent=2))

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"task_generation_real_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_jsonable, f, ensure_ascii=False, indent=2)
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
