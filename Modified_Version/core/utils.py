import yaml
from pprint import pprint
from node_models import AgentRunningState
import jsonlines as jsl
from typing import Any, Dict, List
from loguru import logger

def load_config(config_path: str):
    # Each config file will generate a different scenarios ~1440
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


import itertools


def generate_scenarios(pre_execute: dict) -> list[AgentRunningState]:
    """Generate task-aware scenarios from the pre_execute config block."""
    tasks: List[str] = pre_execute["task"]
    cs_ratios: List[str] = pre_execute["cs_ratio"]

    shared: Dict[str, Any] = pre_execute["shared"]
    char: Dict[str, Any] = shared["character_setting"]

    domains: List[str] = shared["topic"]
    tenses: List[str] = shared["tense"]
    perspectives: List[str] = shared["perspective"]
    genders: List[str] = char["gender"]
    ages: List[str] = char["age"]
    edu_levels: List[str] = char["education_level"]
    conversation_types: List[str] = shared["conversation_type"]
    cs_functions: List[str] = shared["cs_function"]
    cs_types: List[str] = shared["cs_type"]

    use_tools: bool = bool(shared.get("use_tools", False))
    first_language: str = char["nationality"]["first_language"]
    second_language: str = char["nationality"]["second_language"]

    base_scenarios: List[Dict[str, Any]] = []
    for (
        domain,
        tense,
        perspective,
        gender,
        age,
        edu_level,
        cs_ratio,
        conversation_type,
        cs_function,
        cs_type,
    ) in itertools.product(
        domains,
        tenses,
        perspectives,
        genders,
        ages,
        edu_levels,
        cs_ratios,
        conversation_types,
        cs_functions,
        cs_types,
    ):
        base_scenarios.append(
            {
                "topic": domain,
                "tense": tense,
                "perspective": perspective,
                "gender": gender,
                "age": age,
                "education_level": edu_level,
                "cs_ratio": cs_ratio,
                "use_tools": use_tools,
                "conversation_type": conversation_type,
                "first_language": first_language,
                "second_language": second_language,
                "cs_function": cs_function,
                "cs_type": cs_type,
                "data_generation_result": [],
                "response": "",
                "refine_count": 0,
            }
        )

    all_scenarios: List[Dict[str, Any]] = []

    if "topic" in tasks:
        topic_labels: List[str] = pre_execute["topic"]["topics"]
        for base in base_scenarios:
            for lbl in topic_labels:
                scenario = dict(base)
                scenario["task"] = "topic"
                scenario["label"] = lbl
                scenario["topic"] = lbl
                all_scenarios.append(scenario)

    if "sentiment" in tasks:
        sent_cfg: Dict[str, Any] = pre_execute["sentiment"]
        sent_labels: List[str] = sent_cfg["labels"]
        intensities: List[str] = sent_cfg.get("intensity", ["low"])
        ambiguities: List[str] = sent_cfg.get("ambiguity", ["low"])
        for base in base_scenarios:
            for lbl, intensity, ambiguity in itertools.product(
                sent_labels, intensities, ambiguities
            ):
                scenario = dict(base)
                scenario["task"] = "sentiment"
                scenario["label"] = lbl
                scenario["task_constraints"] = {
                    "intensity": intensity,
                    "ambiguity": ambiguity,
                }
                all_scenarios.append(scenario)

    if "ner" in tasks:
        ner_cfg: Dict[str, Any] = pre_execute["ner"]
        entity_types: List[str] = ner_cfg["entity_types"]
        min_entities_list: List[int] = ner_cfg.get("min_entities", [2])
        max_entities_list: List[int] = ner_cfg.get("max_entities", [3])
        must_include_types: List[str] = ner_cfg.get("must_include_types", [])
        allow_cs_entities_list: List[bool] = ner_cfg.get(
            "allow_code_switched_entities", [True]
        )
        for base in base_scenarios:
            for min_e, max_e, allow_cs_entities in itertools.product(
                min_entities_list, max_entities_list, allow_cs_entities_list
            ):
                scenario = dict(base)
                scenario["task"] = "ner"
                scenario["annotations"] = []
                scenario["task_constraints"] = {
                    "entity_types": entity_types,
                    "min_entities": min_e,
                    "max_entities": max_e,
                    "must_include_types": must_include_types,
                    "allow_code_switched_entities": allow_cs_entities,
                }
                all_scenarios.append(scenario)

    return all_scenarios

############################ update to compute ############################
import re

def compute_true_cs_stats(text: str):
    if text is None:
        text = ""
    print('hi ', text)
    ar_tokens = re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', text)
    en_tokens = re.findall(r'[A-Za-z]+', text)

    ar_count = len(ar_tokens)
    en_count = len(en_tokens)
    total = ar_count + en_count

    if total == 0:
        return {
            "cs_ar_count": 0,
            "cs_en_count": 0,
            "cs_ar_ratio": 0.0,
            "cs_en_ratio": 0.0,
            "is_code_switched": False,
        }

    return {
        "cs_ar_count": ar_count,
        "cs_en_count": en_count,
        "cs_ar_ratio": round(ar_count / total, 4)*100,
        "cs_en_ratio": round(en_count / total, 4)*100,
        "is_code_switched": (ar_count > 0 and en_count > 0),
    }

############################ ############################


def weighting_scheme(state):
    fluency_per_instances = state.get("fluency_results_per_instances", [])
    if fluency_per_instances:
        fluency_values = [item.get("fluency_score", 0) for item in fluency_per_instances]
        fluency = sum(fluency_values) / len(fluency_values)
    else:
        fluency = state.get("fluency_result", {}).get("fluency_score", 0)

    naturalness_per_instances = state.get("naturalness_results_per_instances", [])
    if naturalness_per_instances:
        naturalness_values = [
            item.get("naturalness_score", 0) for item in naturalness_per_instances
        ]
        naturalness = sum(naturalness_values) / len(naturalness_values)
    else:
        naturalness = state.get("naturalness_result", {}).get("naturalness_score", 0)

    cs_ratio_results_per_instances = state.get("cs_ratio_results_per_instances", [])
    csRatio_values = [item.get('ratio_score', 0) for item in cs_ratio_results_per_instances if 'ratio_score' in item]

    average_ratio = sum(csRatio_values) / len(csRatio_values) if csRatio_values else 0
    csratio = average_ratio

    socio_per_instances = state.get("social_cultural_results_per_instances", [])
    if socio_per_instances:
        socio_values = [item.get("socio_cultural_score", 0) for item in socio_per_instances]
        socio = sum(socio_values) / len(socio_values)
    else:
        socio = state.get("social_cultural_result", {}).get("socio_cultural_score", 0)

    return fluency * 0.3 + naturalness * 0.25 + csratio * 0.2 + socio * 0.25


def compute_sentence_weighted_scores(state):
    """Compute a weighted score per sentence, aligned by index.

    Missing metric items are treated as 0.0 for that sentence.
    """
    sentences = state.get("data_generation_result", [])
    if not isinstance(sentences, list):
        sentences = [sentences] if sentences is not None else []

    fluency_items = state.get("fluency_results_per_instances", []) or []
    naturalness_items = state.get("naturalness_results_per_instances", []) or []
    cs_ratio_items = state.get("cs_ratio_results_per_instances", []) or []
    socio_items = state.get("social_cultural_results_per_instances", []) or []

    def _get(items, index, key):
        if index < len(items) and isinstance(items[index], dict):
            try:
                return float(items[index].get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    scores = []
    for i in range(len(sentences)):
        fluency = _get(fluency_items, i, "fluency_score")
        naturalness = _get(naturalness_items, i, "naturalness_score")
        csratio = _get(cs_ratio_items, i, "ratio_score")
        socio = _get(socio_items, i, "socio_cultural_score")
        weighted = fluency * 0.3 + naturalness * 0.25 + csratio * 0.2 + socio * 0.25
        scores.append(round(weighted, 6))

    return scores


def build_sentence_records(
    state,
    sentence_scores: list,
    failing_sentence_indices: list,
    refine_counts: list,
    threshold: float = 8.0,
    max_refines: int = 1,
) -> list:
    """Build per-sentence record objects alongside existing parallel arrays.

    This is purely additive — existing arrays in state are never modified.
    Missing array items degrade gracefully to 0.0 / None.

    Status mapping:
        'pass'             — score >= threshold, never refined
        'refined_pass'     — score >= threshold after at least one refinement
        'fail'             — score < threshold, still has refinement budget
        'budget_exhausted' — score < threshold, budget fully used
    """
    texts = state.get("data_generation_result", []) or []
    fluency_items = state.get("fluency_results_per_instances", []) or []
    naturalness_items = state.get("naturalness_results_per_instances", []) or []
    cs_ratio_items = state.get("cs_ratio_results_per_instances", []) or []
    socio_items = state.get("social_cultural_results_per_instances", []) or []

    # ── Alignment guard: warn loudly if any metric array is shorter than sentence count ──
    n = len(texts)
    for name, arr in [
        ("fluency_results_per_instances",          fluency_items),
        ("naturalness_results_per_instances",      naturalness_items),
        ("cs_ratio_results_per_instances",         cs_ratio_items),
        ("social_cultural_results_per_instances",  socio_items),
    ]:
        if len(arr) != n:
            logger.warning(
                f"build_sentence_records: array length mismatch — "
                f"{name} has {len(arr)} items but data_generation_result has {n}. "
                f"Sentences at missing indices will have empty metric objects."
            )

    task_validation_result = state.get("task_validation_result") or {}
    per_instance_tv = (
        task_validation_result.get("per_instance_results", []) or []
        if isinstance(task_validation_result, dict)
        else []
    )

    # Normalise refine_counts to match text count
    rc = list(refine_counts) if isinstance(refine_counts, list) else []
    while len(rc) < len(texts):
        rc.append(0)

    failing_set = (
        set(failing_sentence_indices)
        if isinstance(failing_sentence_indices, list)
        else set()
    )

    def _get_item(arr, idx):
        """Return the full response dict at arr[idx], or an empty dict if missing."""
        if idx < len(arr) and isinstance(arr[idx], dict):
            return arr[idx]
        return {}

    records = []
    for i, text in enumerate(texts):
        score = float(sentence_scores[i]) if i < len(sentence_scores) else 0.0
        rc_i = int(rc[i]) if i < len(rc) else 0
        is_failing = i in failing_set

        if not is_failing and rc_i == 0:
            status = "pass"
        elif not is_failing and rc_i > 0:
            status = "refined_pass"
        elif is_failing and rc_i >= max_refines:
            status = "budget_exhausted"
        else:
            status = "fail"

        # Per-sentence task validation flag
        task_passed = None
        tv_item = per_instance_tv[i] if i < len(per_instance_tv) else None
        if isinstance(tv_item, dict):
            task_passed = bool(tv_item.get("passed", False))
        elif not per_instance_tv and isinstance(task_validation_result, dict):
            raw = task_validation_result.get("passed")
            task_passed = bool(raw) if raw is not None else None

        # Store the full typed response objects, not just extracted floats
        records.append(
            {
                "index": i,
                "text": text,
                "fluency": _get_item(fluency_items, i),           # full FluencyResponse
                "naturalness": _get_item(naturalness_items, i),   # full NaturalnessResponse
                "cs_ratio": _get_item(cs_ratio_items, i),         # full CSRatioResponse
                "socio_cultural": _get_item(socio_items, i),      # full SocialCulturalResponse
                "weighted_score": round(score, 6),
                "refine_count": rc_i,
                "status": status,
                "task_passed": task_passed,
            }
        )

    return records


if __name__ == "__main__":
    # Here is your config dictionary (simplified for the example):
    config = {
        "character_setting": {
            "age": ["8-17", "18-25", "26-35", "56-65", "66+"],
            "education_level": ["High School", "College", "Master", "Doctor"],
            "gender": ["Male", "Female"],
            "nationality": {
                "first_language": "Cantonese",
                "second_language": "English",
            },
        },
        "cs_function": [
            "Directive",
            "Expressive",
            "Referential",
            "Phatic",
            "Metalinguistic",
            "Poetic",   
        ],
        "cs_type": [
            "Intersentential",
            "Intrasentential",
            "Extra-sentential / Tag switching",
        ],
        "cs_ratio": ["Low", "Medium", "High"],
        "output_format": "json",
        "output_type": "single_turn",
        "perspective": ["First Person", "Third Person"],
        "tense": ["Past", "Present", "Future"],
        "topics": ["Tourism", "Weather", "Shopping", "Food", "Exam", "Politics"],
        "use_tools": True,
        "conversation_type": ["single_turn", "multi-turn"],
    }

    scenarios = generate_scenarios(config)
    print(f"Generated {len(scenarios)} scenario combinations.")
    # For a quick peek, let's print the first few
    for i, sc in enumerate(scenarios[:10]):
        print(f"Scenario #{i+1}:", sc)
        print("\n")
