import os
import re
import json
from pathlib import Path
from utils import compute_true_cs_stats

import dotenv
import random
import jsonlines
from langchain_openai import ChatOpenAI
from prompt import (
    DATA_GENERATION_PROMPT,
    DATA_GENERATION_TOPIC_PROMPT,
    DATA_GENERATION_SENTIMENT_PROMPT,
    DATA_GENERATION_NER_PROMPT,
    TASK_VALIDATION_TOPIC_PROMPT,
    TASK_VALIDATION_SENTIMENT_PROMPT,
    TASK_VALIDATION_NER_PROMPT,
    FLUENCY_PROMPT,
    NATURALNESS_PROMPT,
    CS_RATIO_PROMPT,
    SOCIAL_CULTURAL_PROMPT,
    REFINER_PROMPT,
    REFINER_TASK_TOPIC_PROMPT,
    REFINER_TASK_SENTIMENT_PROMPT,
    REFINER_TASK_NER_PROMPT,
)
from node_models import (
    AgentRunningState,
    GenerationResponse,
    FluencyResponse,
    NaturalnessResponse,
    CSRatioResponse,
    SocialCulturalResponse,
    TaskValidationResult,
)
from utils import (
    weighting_scheme,
    compute_sentence_weighted_scores,
    build_sentence_records,
    validate_sentence_records_consistency,
)
from copy import deepcopy

from mcp_tools import get_all_tools
from typing import Dict, Any
# from google.colab import userdata

# Load the intended env file first to avoid picking stale keys from sibling folders.
_CURRENT_FILE = Path(__file__).resolve()
_ENV_CANDIDATES = [
    _CURRENT_FILE.parents[1] / ".env",  # drive_code/.env (preferred)
    _CURRENT_FILE.parent / ".env",      # drive_code/core/.env (legacy)
    _CURRENT_FILE.parents[2] / ".env",  # workspace root .env (fallback)
]
for _env_path in _ENV_CANDIDATES:
    if _env_path.exists():
        dotenv.load_dotenv(dotenv_path=_env_path, override=True)
        break

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE = os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE")
MODEL = "gpt-4o-mini"
OUTPUT_DIR = str(_CURRENT_FILE.parents[1] / "output")
# Sentence-level cap: max refine attempts allowed per individual sentence.
MAX_SENTENCE_REFINES = int(os.getenv("MAX_SENTENCE_REFINES", "1"))

# Task-aware acceptance (default ON, like the other env knobs above).
# When "1", AcceptanceAgent writes ONLY sentences whose FINAL task verdict passed
# (sentence_records[i]["task_passed"] is not False); sentences the refiner could not
# make task-correct are dropped from the written corpus instead of accepted as
# budget_exhausted. Set TASK_AWARE_ACCEPT=0 to reproduce the historical corpora
# (240/480/960/V1 sentiment sets were generated with this behaviour OFF).
TASK_AWARE_ACCEPT = os.getenv("TASK_AWARE_ACCEPT", "1").strip() == "1"

# assert API_KEY is not None, "OPENAI_API_KEY is not set"


def RunSampleAgent(state: AgentRunningState):
    SampleAgent = SAMPLE_AGENT_PROMPT | ChatOpenAI(
        model=MODEL, temperature=1, base_url=API_BASE, api_key=API_KEY
    ).with_structured_output(GenerationResponse)
    response = SampleAgent.invoke(state)
    if response.get("type"):
        return {"response": ""}
    # print(response.content)
    # copy the state and add the responsev
    payload = state.copy()
    payload["response"] = response
    with jsonlines.open("result/simple_agent_result_new.jsonl", "a") as f:
        f.write(response)
    return {"response": response}


def RunUseToolsAgent(state: AgentRunningState):
    UseToolsAgent = USE_TOOLS_PROMPT | ChatOpenAI(
        model=MODEL, temperature=1, base_url=API_BASE, api_key=API_KEY
    ).with_structured_output(GenerationResponse)
    random_news = []
    with jsonlines.open("news/news_data_till241201.jsonl") as f:
        for line in f:
            random_news.append(line)

    random_news = random.sample(random_news, 1)
    state["news_article"] = random_news[0]["title"] + "\n" + random_news[0]["content"]
    response = UseToolsAgent.invoke(state)
    payload = deepcopy(state)
    del payload["topic"]
    try:
        payload["news_generation_result"] = response["instances"]
        with jsonlines.open("result/use_tools_result_new.jsonl", "a") as f:
            f.write(payload)
    except Exception as e:
        print(response)
    return {"news_generation_result": response["instances"]}


# def RunDataGenerationAgent(state: AgentRunningState):
#     state["mcp_result"] = ""
#     print('state',state)

#     if state.get("topic") not in state["news_dict"]:
#         state["news_article"] = ""
#     else:
#         if state.get("topic") in state["news_hash"]:
#             state["news_article"] = random.choice(state["news_dict"][state["topic"]])
#         else:
#             state["news_article"] = random.choice(state["news_dict"][state["topic"]])
#             state["news_hash"].add(state["topic"])
#     DataGenerationAgent = DATA_GENERATION_PROMPT | ChatOpenAI(
#         model=MODEL, temperature=0.7, base_url=API_BASE
#     ).with_structured_output(GenerationResponse)
#     response = DataGenerationAgent.invoke(state)
#     retry = 4
#     if not response.get("instances"):
#         while retry > 0:
#             response = DataGenerationAgent.invoke(state)
#             if response.get("instances"):
#                 break
#             retry -= 1
#     return {"data_generation_result": response["instances"]}

###################### update run generation #######################
def _invoke_generation_with_retry(state: AgentRunningState, generation_prompt):
    generation_agent = generation_prompt | ChatOpenAI(
        model=MODEL, temperature=0.7, base_url=API_BASE, api_key=API_KEY
    ).with_structured_output(GenerationResponse)

    response = generation_agent.invoke(state)
    retry = 4
    if not response.get("instances"):
        while retry > 0:
            response = generation_agent.invoke(state)
            if response.get("instances"):
                break
            retry -= 1
    return response


def RunTopicDataGenerationAgent(state: AgentRunningState):
    return _invoke_generation_with_retry(state, DATA_GENERATION_TOPIC_PROMPT)


def RunSentimentDataGenerationAgent(state: AgentRunningState):
    return _invoke_generation_with_retry(state, DATA_GENERATION_SENTIMENT_PROMPT)


# --- Generic, config-driven NER entity guidance -----------------------------------------
# Per-type metadata in ONE place (overridable by task_constraints["entity_type_guidance"]).
# Add a new tag here (or in config) — never in the prompt text.
DEFAULT_ENTITY_GUIDANCE = {
    "PER":     {"description": "a named person, preferably a full first and last name",
                "script_rule": "must be written in English/Latin letters",
                "examples": ["Ahmed Ali", "Sarah Hassan", "Omar Khaled", "Mona Ibrahim", "Elon Musk"]},
    "ORG":     {"description": "a company, university, institution, brand, or organization",
                "script_rule": "must be written in English/Latin letters",
                "examples": ["Google", "Microsoft", "Apple", "Cairo University"]},
    "LOC":     {"description": "a city, country, district, landmark, or place",
                "script_rule": "must be written in English/Latin letters",
                "examples": ["Cairo", "Dubai", "London", "New York"]},
    "PRODUCT": {"description": "a named product, app, software, platform, device, or service",
                "script_rule": "must be written in English/Latin letters",
                "examples": ["iPhone 15", "ChatGPT", "Windows", "Photoshop"]},
    "EVENT":   {"description": "a named event, conference, tournament, festival, campaign, or public occasion",
                "script_rule": "must be written in English/Latin letters",
                "examples": ["World Cup", "Olympics", "Black Friday", "Cairo Book Fair", "Gitex Global"]},
}


def build_ner_entity_guidance(task_constraints) -> str:
    """Build a dynamic guidance block for ONLY the required entity types, from per-type metadata.
    Source priority: task_constraints['entity_type_guidance'][TYPE] -> DEFAULT_ENTITY_GUIDANCE[TYPE]
    -> generic fallback. Script rule defaults to target_entities_script (english -> Latin only)."""
    c = task_constraints or {}
    def _scalar(v, d):
        v = v[0] if isinstance(v, list) and v else v
        return d if v is None else v
    must = [str(t).strip().upper() for t in (c.get("must_include_types") or [])]
    types = must or [str(t).strip().upper() for t in (c.get("entity_types") or [])]
    script = str(_scalar(c.get("target_entities_script"), "english")).strip().lower()
    cfg_guidance = c.get("entity_type_guidance") or {}
    default_rule = ("must be written in English/Latin letters" if script == "english"
                    else "may be written in Arabic or English script")
    lines = []
    for t in types:
        g = cfg_guidance.get(t) or cfg_guidance.get(t.lower()) or DEFAULT_ENTITY_GUIDANCE.get(t) or {}
        desc = g.get("description", f"a named entity of type {t}")
        rule = g.get("script_rule", default_rule)
        ex = g.get("examples") or []
        ex_str = f" Examples: {', '.join(str(e) for e in ex[:5])}." if ex else ""
        lines.append(f"- {t}: {desc}; {rule}.{ex_str}")
    return "Required entity guidance:\n" + "\n".join(lines) if lines else "No specific entity types required."


def RunNERDataGenerationAgent(state: AgentRunningState):
    # Inject dynamic per-type guidance for the required entity types (fills {ner_entity_guidance}).
    # Config-driven: scenario-level entity_type_guidance (from YAML) overrides DEFAULT_ENTITY_GUIDANCE.
    state = dict(state)
    cons = dict(state.get("task_constraints", {}) or {})
    if state.get("entity_type_guidance"):
        cons.setdefault("entity_type_guidance", state["entity_type_guidance"])
    state["ner_entity_guidance"] = build_ner_entity_guidance(cons)
    return _invoke_generation_with_retry(state, DATA_GENERATION_NER_PROMPT)


def _invoke_task_validation_with_retry(state: AgentRunningState, validator_prompt):
    validator_agent = validator_prompt | ChatOpenAI(
        model=MODEL, temperature=0.1, base_url=API_BASE, api_key=API_KEY
    ).with_structured_output(TaskValidationResult)

    response = validator_agent.invoke(state)
    retry = 2
    if response is None:
        while retry > 0:
            response = validator_agent.invoke(state)
            if response is not None:
                break
            retry -= 1
    return response or {
        "passed": False,
        "confidence": 0.0,
        "notes": "Validator returned empty response",
        "predicted_label": None,
        "errors": ["empty_validator_response"],
    }


def _validate_per_instance_with_retry(state: AgentRunningState, validator_prompt) -> Dict[str, Any]:
    instances = state.get("data_generation_result", [])
    if not isinstance(instances, list) or not instances:
        aggregate = _invoke_task_validation_with_retry(state, validator_prompt)
        return {
            "aggregate": aggregate,
            "per_instance_results": [],
        }

    per_instance_results = []
    all_errors: list[str] = []
    confidence_values: list[float] = []
    predicted_labels: list[str] = []

    for index, text in enumerate(instances):
        single_state = dict(state)
        single_state["data_generation_result"] = [text]

        result = _invoke_task_validation_with_retry(single_state, validator_prompt)
        if isinstance(result, dict):
            per_instance_results.append(result)
            confidence_values.append(float(result.get("confidence", 0.0)))
            predicted_label = result.get("predicted_label")
            if isinstance(predicted_label, str) and predicted_label.strip():
                predicted_labels.append(predicted_label.strip())
            for error in result.get("errors", []) or []:
                all_errors.append(f"instance_{index}: {error}")

    if not per_instance_results:
        return {
            "aggregate": {
            "passed": False,
            "confidence": 0.0,
            "notes": "Validator returned empty per-instance responses",
            "predicted_label": None,
            "errors": ["empty_per_instance_validator_response"],
            },
            "per_instance_results": [],
        }

    passed = all(bool(item.get("passed", False)) for item in per_instance_results)
    confidence = round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else 0.0
    unique_predicted = sorted(set(predicted_labels))
    if len(unique_predicted) == 1:
        aggregate_predicted = unique_predicted[0]
    elif len(unique_predicted) > 1:
        aggregate_predicted = "mixed"
    else:
        aggregate_predicted = None

    notes = "Per-instance task validation"
    if unique_predicted:
        notes += f"; predicted_labels={unique_predicted}"

    return {
        "aggregate": {
            "passed": passed,
            "confidence": confidence,
            "notes": notes,
            "predicted_label": aggregate_predicted,
            "errors": all_errors,
        },
        "per_instance_results": per_instance_results,
    }


def RunTopicTaskValidatorAgent(state: AgentRunningState):
    response = _validate_per_instance_with_retry(state, TASK_VALIDATION_TOPIC_PROMPT)
    return {
        "task_validation_result": response.get("aggregate", {}),
        "task_validation_results_per_instances": response.get("per_instance_results", []),
    }


def RunSentimentTaskValidatorAgent(state: AgentRunningState):
    response = _validate_per_instance_with_retry(state, TASK_VALIDATION_SENTIMENT_PROMPT)
    return {
        "task_validation_result": response.get("aggregate", {}),
        "task_validation_results_per_instances": response.get("per_instance_results", []),
    }


def _normalize_entity_type(entity_type: str) -> str:
    normalized = str(entity_type).strip().upper()
    alias_map = {
        "PERSON": "PER",
        "PEOPLE": "PER",
        "ORGANIZATION": "ORG",
        "ORGANISATION": "ORG",
        "LOCATION": "LOC",
        "PLACE": "LOC",
        "GPE": "LOC",
    }
    return alias_map.get(normalized, normalized)


def _extract_english_ner_counts(text: str, requested_entity_types: list[str]) -> Dict[str, int]:
    base_counts = {"PER": 0, "ORG": 0, "LOC": 0}

    ascii_text = re.sub(r"[^A-Za-z0-9&\.\-\'\s]", " ", text)
    ascii_text = re.sub(r"\s+", " ", ascii_text).strip()

    per_candidates = set()
    for match in re.finditer(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", text):
        start, end = match.span(1)
        prev_word_match = re.search(r"([A-Za-z]+)\s*$", text[:start])
        next_word_match = re.match(r"^\s*([A-Za-z]+)", text[end:])

        prev_is_title = bool(
            prev_word_match and re.fullmatch(r"[A-Z][a-z]+", prev_word_match.group(1))
        )
        next_is_title = bool(
            next_word_match and re.fullmatch(r"[A-Z][a-z]+", next_word_match.group(1))
        )

        if not prev_is_title and not next_is_title:
            per_candidates.add(match.group(1))
    non_person_tail_tokens = {
        "Docs", "News", "Bank", "University", "Company", "Corporation", "Institute", "Labs", "Lab", "Tech",
    }
    per_entities = {
        item for item in per_candidates if item.split()[-1] not in non_person_tail_tokens
    }

    org_entities = set()
    org_entities.update(
        re.findall(
            r"\b([A-Z][A-Za-z0-9&\.-]*(?:\s+[A-Z][A-Za-z0-9&\.-]*){0,2}\s+"
            r"(?:Inc|Corp|Corporation|Ltd|LLC|Group|University|Bank|Agency|Company|Institute|Labs|Lab|Technologies|Tech))\b",
            ascii_text,
        )
    )
    org_entities.update(
        re.findall(
            r"\b(?:company|organization|institution|bank|Ø¬Ø§Ù…Ø¹Ø©|Ø´Ø±ÙƒØ©|Ø¨Ù†Ùƒ)\s+([A-Z][A-Za-z0-9&\.-]*(?:\s+[A-Z][A-Za-z0-9&\.-]*)?)\b",
            ascii_text,
            flags=re.IGNORECASE,
        )
    )
    org_entities.update(
        re.findall(
            r"\b(?:program|app|application|platform|tool|service|product|Ø¨Ø±Ù†Ø§Ù…Ø¬|ØªØ·Ø¨ÙŠÙ‚|Ù…Ù†ØµØ©|Ø®Ø¯Ù…Ø©|Ù…Ù†ØªØ¬)\s+([A-Z][A-Za-z0-9&\.-]*(?:\s+[A-Z][A-Za-z0-9&\.-]*)?)\b",
            ascii_text,
            flags=re.IGNORECASE,
        )
    )
    org_entities.update(
        re.findall(
            r"\b(?:from|by|via|using|use|about|with|Ù…Ù†|Ø¹Ù†|Ø¨Ø§Ø³ØªØ®Ø¯Ø§Ù…)\s+([A-Z][A-Za-z0-9&\.-]*(?:\s+[A-Z][A-Za-z0-9&\.-]*)?)\b",
            ascii_text,
            flags=re.IGNORECASE,
        )
    )

    loc_entities = set()
    loc_entities.update(
        re.findall(
            r"\b(?:in|at|from|to|into|near|around|inside|outside)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
            ascii_text,
            flags=re.IGNORECASE,
        )
    )
    loc_entities.update(
        re.findall(r"\bÙÙŠ\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text)
    )

    standalone_single_tokens = set(re.findall(r"\b([A-Z][a-z]{2,})\b", ascii_text))
    covered_tokens = set()
    for ent in per_entities.union(org_entities).union(loc_entities):
        for token in ent.split():
            covered_tokens.add(token)

    fallback_org = standalone_single_tokens - covered_tokens
    org_entities.update(fallback_org)

    org_tokens = {token for ent in org_entities for token in ent.split()}
    loc_entities = {ent for ent in loc_entities if all(tok not in org_tokens for tok in ent.split())}

    base_counts["PER"] = len(per_entities)
    base_counts["ORG"] = len(org_entities)
    base_counts["LOC"] = len(loc_entities)

    requested_normalized = [_normalize_entity_type(entity_type) for entity_type in requested_entity_types]
    if not requested_normalized:
        return base_counts

    dynamic_counts: Dict[str, int] = {}
    for entity_type in requested_normalized:
        dynamic_counts[entity_type] = base_counts.get(entity_type, 0)
    return dynamic_counts


def _deterministic_ner_english_policy(state: AgentRunningState) -> Dict[str, Any]:
    instances = state.get("data_generation_result", [])
    combined = " ".join(instances) if isinstance(instances, list) else ""

    constraints = state.get("task_constraints", {}) if isinstance(state.get("task_constraints", {}), dict) else {}
    min_entities = int(constraints.get("min_entities", 0))
    max_entities = int(constraints.get("max_entities", 10**9))
    required_entity_types_raw = constraints.get("entity_types", [])
    must_types_raw = constraints.get("must_include_types", [])

    required_entity_types = [_normalize_entity_type(entity_type) for entity_type in required_entity_types_raw]
    must_types = [_normalize_entity_type(entity_type) for entity_type in must_types_raw]

    counts = _extract_english_ner_counts(combined, required_entity_types)

    total_entities = sum(counts.get(entity_type, 0) for entity_type in required_entity_types)
    has_all_entity_types = all(counts.get(entity_type, 0) > 0 for entity_type in required_entity_types)
    has_required_types = all(counts.get(entity_type, 0) > 0 for entity_type in must_types)
    within_range = min_entities <= total_entities <= max_entities
    passed = (
        bool(instances)
        and has_all_entity_types
        and has_required_types
        and within_range
    )

    required_type_coverage = (
        sum(1 for entity_type in required_entity_types if counts.get(entity_type, 0) > 0) / len(required_entity_types)
        if required_entity_types
        else 1.0
    )
    must_type_coverage = (
        sum(1 for entity_type in must_types if counts.get(entity_type, 0) > 0) / len(must_types)
        if must_types
        else 1.0
    )
    has_instances_score = 1.0 if instances else 0.0
    within_range_score = 1.0 if within_range else 0.0
    confidence = round(
        (
            0.35 * required_type_coverage
            + 0.35 * must_type_coverage
            + 0.20 * within_range_score
            + 0.10 * has_instances_score
        ),
        4,
    )

    errors = []
    if not instances:
        errors.append("No generated instances.")
    if not has_all_entity_types:
        errors.append(f"All English entity_types not satisfied: {required_entity_types}")
    if not has_required_types:
        errors.append(f"Required English entity types not satisfied: {must_types}")
    if not within_range:
        errors.append(
            f"English entity count {total_entities} is outside range [{min_entities}, {max_entities}]"
        )

    return {
        "passed": passed,
        "confidence": confidence,
        "notes": "Deterministic English-entity policy check",
        "predicted_label": None,
        "errors": errors,
        "english_entity_counts": counts,
        "english_total_entities": total_entities,
    }


def RunNERTaskValidatorAgent(state: AgentRunningState):
    llm_response = _validate_per_instance_with_retry(state, TASK_VALIDATION_NER_PROMPT)
    llm_aggregate = llm_response.get("aggregate", {}) if isinstance(llm_response, dict) else {}
    llm_per_instance = llm_response.get("per_instance_results", []) if isinstance(llm_response, dict) else []
    # deterministic = _deterministic_ner_english_policy(state)
    llm_notes = llm_aggregate.get("notes", "") if isinstance(llm_aggregate, dict) else ""

    final_result = {
        "passed": bool(llm_aggregate.get("passed", False)) if isinstance(llm_aggregate, dict) else False,
        "confidence": float(llm_aggregate.get("confidence", 0.0)) if isinstance(llm_aggregate, dict) else 0.0,
        "notes": llm_notes,
        "deterministic_notes": "",
        "llm_notes": llm_notes,
        "predicted_label": llm_aggregate.get("predicted_label") if isinstance(llm_aggregate, dict) else None,
        "errors": llm_aggregate.get("errors", []) if isinstance(llm_aggregate, dict) else ["empty_validator_response"],
    }

    # To re-enable hybrid validation later, restore deterministic merge logic:
    # final_result = {
    #     "passed": deterministic["passed"],
    #     "confidence": deterministic["confidence"],
    #     "notes": deterministic["notes"],
    #     "deterministic_notes": deterministic["notes"],
    #     "llm_notes": llm_notes,
    #     "predicted_label": None,
    #     "errors": deterministic["errors"],
    #     "english_entity_counts": deterministic["english_entity_counts"],
    #     "english_total_entities": deterministic["english_total_entities"],
    # }
    return {
        "task_validation_result": final_result,
        "task_validation_results_per_instances": llm_per_instance,
    }


def RunTaskValidatorAgent(state: AgentRunningState):
    task = state.get("task", "topic")
    if task == "sentiment":
        return RunSentimentTaskValidatorAgent(state)
    if task == "ner":
        return RunNERTaskValidatorAgent(state)
    return RunTopicTaskValidatorAgent(state)


def RunDataGenerationAgent(state: AgentRunningState):
    state["mcp_result"] = ""
    print("state", state)

    if state.get("topic") not in state["news_dict"]:
        state["news_article"] = ""
    else:
        if state.get("topic") in state["news_hash"]:
            state["news_article"] = random.choice(state["news_dict"][state["topic"]])
        else:
            state["news_article"] = random.choice(state["news_dict"][state["topic"]])
            state["news_hash"].add(state["topic"])

    task = state.get("task", "topic")
    if task == "sentiment":
        response = RunSentimentDataGenerationAgent(state)
    elif task == "ner":
        response = RunNERDataGenerationAgent(state)
    else:
        response = RunTopicDataGenerationAgent(state)

    instances = response.get("instances", [])
    if not instances:
        raise RuntimeError("DataGenerationAgent produced no instances after retries")

    # # NEW PART STARTS HERE
    # per_instance_stats = [compute_true_cs_stats(x) for x in instances]

    # # instances is usually a list of strings
    # if isinstance(instances, list):
    #     text_for_stats = " ".join(instances)
    # else:
    #     text_for_stats = str(instances)
    # print('text for stats',text_for_stats)
    # cs_stats = compute_true_cs_stats(text_for_stats)
    # print("CS STATS:", cs_stats)

    # # NEW PART ENDS HERE

    return {
        "data_generation_result": response["instances"],  # unchanged key
        # **cs_stats                              # inject deterministic CS info
        #  "cs_stats_per_instance": per_instance_stats,  # NEW
    }
#########################################################################


def RunFluencyAgent(state: AgentRunningState):
    texts = state.get("data_generation_result", [])
    if not texts:
        return {
            "fluency_results_per_instances": [],
            "fluency_result": {
                "fluency_score": 0.0,
                "errors": {},
                "summary": "No instances to evaluate.",
            },
        }

    llm = FLUENCY_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, base_url=API_BASE, api_key=API_KEY
    )

    results = []
    for text in texts:
        response = llm.invoke({"sentence": text})
        item = _extract_json_object(response)
        errors = item.get("errors", {})
        if not isinstance(errors, dict):
            if isinstance(errors, list):
                errors = {str(i + 1): str(err) for i, err in enumerate(errors)}
            else:
                errors = {}
        results.append(
            {
                "fluency_score": _safe_score(item.get("fluency_score")),
                "errors": errors,
                "summary": str(item.get("summary", "")),
            }
        )

    average_score = sum(x["fluency_score"] for x in results) / len(results) if results else 0.0
    aggregate_errors = {
        f"sentence_{idx + 1}": str(item.get("errors", {}))
        for idx, item in enumerate(results)
        if item.get("errors")
    }

    return {
        "fluency_results_per_instances": results,
        "fluency_result": {
            "fluency_score": average_score,
            "errors": aggregate_errors,
            "summary": f"Per-instance average across {len(results)} sentence(s).",
        },
    }


def RunNaturalnessAgent(state: AgentRunningState):
    texts = state.get("data_generation_result", [])
    if not texts:
        return {
            "naturalness_results_per_instances": [],
            "naturalness_result": {
                "naturalness_score": 0.0,
                "observations": {},
                "summary": "No instances to evaluate.",
            },
        }

    llm = NATURALNESS_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, base_url=API_BASE, api_key=API_KEY
    )

    results = []
    for text in texts:
        response = llm.invoke({"sentence": text})
        item = _extract_json_object(response)
        observations = item.get("observations", {})
        if not isinstance(observations, dict):
            if isinstance(observations, list):
                observations = {str(i + 1): str(obs) for i, obs in enumerate(observations)}
            else:
                observations = {}
        results.append(
            {
                "naturalness_score": _safe_score(item.get("naturalness_score")),
                "observations": observations,
                "summary": str(item.get("summary", "")),
            }
        )

    average_score = sum(x["naturalness_score"] for x in results) / len(results) if results else 0.0
    aggregate_observations = {
        f"sentence_{idx + 1}": str(item.get("observations", {}))
        for idx, item in enumerate(results)
        if item.get("observations")
    }

    return {
        "naturalness_results_per_instances": results,
        "naturalness_result": {
            "naturalness_score": average_score,
            "observations": aggregate_observations,
            "summary": f"Per-instance average across {len(results)} sentence(s).",
        },
    }




################################################################
def parse_target_ratios(cs_ratio: str):
    if not cs_ratio:
        return 0.0, 0.0
    try:
        target_matrix = float(cs_ratio.replace("%", ""))
        target_embedded = 100 - target_matrix
        return round(target_matrix, 4), round(target_embedded, 4)
    except (ValueError, AttributeError) as e:
        print(f"ERROR parsing cs_ratio '{cs_ratio}': {e}")
        return 0.0, 0.0


def _is_arabic_language(language: str) -> bool:
    normalized = str(language or "").strip().lower()
    return normalized in {"arabic", "ar"} or "arab" in normalized

def set_field(obj, key, value):
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)

def get_field(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _build_sentences_for_batch(texts: list[str]) -> str:
    return "\n".join([f"Sentence {idx + 1}: {text}" for idx, text in enumerate(texts)])


def _extract_json_array(response: Any) -> list[dict]:
    try:
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip()
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end > start:
            parsed = json.loads(content[start:end])
        else:
            parsed = json.loads(content)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        print(f"ERROR parsing batch response: {e}")
    return []


def _extract_json_object(response: Any) -> dict:
    """Best-effort extraction of a JSON object from an LLM response."""
    try:
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(content[start:end])
        else:
            parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        print(f"ERROR parsing JSON object response: {e}")
    return {}


def _safe_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_batch_dict_results(raw_results: Any, expected_len: int, agent_name: str) -> list[dict]:
    """Normalize evaluator batch results to exactly expected_len dict items."""
    if not isinstance(raw_results, list):
        raw_results = []

    normalized = [item if isinstance(item, dict) else {} for item in raw_results]

    if len(normalized) != expected_len:
        print(
            f"WARNING [{agent_name}]: expected {expected_len} batch items, got {len(normalized)}. "
            "Applying pad/trim normalization."
        )

    if len(normalized) < expected_len:
        normalized.extend([{} for _ in range(expected_len - len(normalized))])
    elif len(normalized) > expected_len:
        normalized = normalized[:expected_len]

    return normalized


def _recover_results_per_sentence(
    texts: list[str],
    invoke_single,
    agent_name: str,
) -> list[dict]:
    """Recover per-sentence evaluator outputs when batch output is unsafe."""
    recovered: list[dict] = []
    print(f"WARNING [{agent_name}]: entering per-sentence fallback recovery for {len(texts)} sentence(s).")

    for idx, text in enumerate(texts):
        try:
            response = invoke_single(idx, text)
            parsed_arr = _extract_json_array(response)
            if parsed_arr:
                item = parsed_arr[0] if isinstance(parsed_arr[0], dict) else {}
            else:
                item = _extract_json_object(response)
            if not isinstance(item, dict):
                item = {}
            recovered.append(item)
        except Exception as e:
            print(f"WARNING [{agent_name}]: single-sentence fallback failed at index {idx}: {e}")
            recovered.append({})

    return recovered


def RunCSRatioAgent(state: AgentRunningState):
    texts = state.get("data_generation_result", [])
    if not texts:
        return {"cs_ratio_results_per_instances": []}

    target_matrix, target_embedded = parse_target_ratios(state.get("cs_ratio"))
    matrix_is_arabic = _is_arabic_language(state.get("first_language", ""))

    llm = CS_RATIO_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, base_url=API_BASE, api_key=API_KEY
    )

    results = []
    for text in texts:
        stats = compute_true_cs_stats(text)
        mapped = {
            "matrix_ratio": stats.get("cs_ar_ratio", 0.0) if matrix_is_arabic else stats.get("cs_en_ratio", 0.0),
            "embedded_ratio": stats.get("cs_en_ratio", 0.0) if matrix_is_arabic else stats.get("cs_ar_ratio", 0.0),
            "matrix_count": stats.get("cs_ar_count", 0) if matrix_is_arabic else stats.get("cs_en_count", 0),
            "embedded_count": stats.get("cs_en_count", 0) if matrix_is_arabic else stats.get("cs_ar_count", 0),
        }
        sentence_with_stats = (
            f"{text}\n"
            f"  - Matrix ratio: {mapped['matrix_ratio']:.2f}%\n"
            f"  - Embedded ratio: {mapped['embedded_ratio']:.2f}%\n"
            f"  - Matrix tokens: {mapped['matrix_count']}\n"
            f"  - Embedded tokens: {mapped['embedded_count']}\n"
            f"  - Is code-switched: {stats['is_code_switched']}\n"
        )
        response = llm.invoke({
            "cs_ratio": state.get("cs_ratio"),
            "target_matrix_ratio": target_matrix,
            "target_embedded_ratio": target_embedded,
            "sentence_with_stats": sentence_with_stats,
        })
        item = _extract_json_object(response)
        results.append({
            "ratio_score": _safe_score(item.get("ratio_score")),
            "computed_ratio": str(item.get("computed_ratio", f"{mapped['matrix_ratio']:.2f}% : {mapped['embedded_ratio']:.2f}%")),
            "notes": str(item.get("notes", "monolingual" if not stats["is_code_switched"] else "code-switched")),
        })

    return {"cs_ratio_results_per_instances": results}






def RunSocialCulturalAgent(state: AgentRunningState):
    texts = state.get("data_generation_result", [])
    if not texts:
        return {
            "social_cultural_results_per_instances": [],
            "social_cultural_result": {
                "socio_cultural_score": 0.0,
                "issues": "",
                "summary": "No instances to evaluate.",
            },
        }

    llm = SOCIAL_CULTURAL_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, base_url=API_BASE, api_key=API_KEY
    )

    results = []
    for text in texts:
        response = llm.invoke({"sentence": text})
        item = _extract_json_object(response)
        issues = item.get("issues", "")
        if isinstance(issues, list):
            issues = "; ".join([str(x) for x in issues])
        results.append(
            {
                "socio_cultural_score": _safe_score(item.get("socio_cultural_score")),
                "issues": str(issues),
                "summary": str(item.get("summary", "")),
            }
        )

    average_score = sum(x["socio_cultural_score"] for x in results) / len(results) if results else 0.0
    aggregate_issues = " | ".join(
        [f"sentence_{idx + 1}: {item['issues']}" for idx, item in enumerate(results) if item.get("issues")]
    )

    return {
        "social_cultural_results_per_instances": results,
        "social_cultural_result": {
            "socio_cultural_score": average_score,
            "issues": aggregate_issues,
            "summary": f"Per-instance average across {len(results)} sentence(s).",
        },
    }


def SummarizeResult(state: AgentRunningState):
    sentence_scores = compute_sentence_weighted_scores(state)
    sentence_threshold = 8.0
    failing_sentence_indices = [
        i for i, score in enumerate(sentence_scores) if float(score) < sentence_threshold
    ]

    refine_counts = state.get("instance_refine_counts", [])
    if not isinstance(refine_counts, list):
        refine_counts = []
    if len(refine_counts) < len(state.get("data_generation_result", [])):
        refine_counts = refine_counts + [0] * (
            len(state.get("data_generation_result", [])) - len(refine_counts)
        )

    summary = f"""
    data_generation_result: {state["data_generation_result"]}
    Sentence Scores: {sentence_scores}
    Failing Sentence Indices (<{sentence_threshold}): {failing_sentence_indices}
    Fluency Per-Instance: {state.get("fluency_results_per_instances", [])}
    Fluency Result: {state["fluency_result"]}
    Naturalness Per-Instance: {state.get("naturalness_results_per_instances", [])}
    Naturalness Result: {state["naturalness_result"]}
    CSRatio Result: {state["cs_ratio_results_per_instances"]}
    Social Cultural Per-Instance: {state.get("social_cultural_results_per_instances", [])}
    Social Cultural Result: {state["social_cultural_result"]}
    """
    state["summary"] = summary
    # print(summary)
    # with jsonlines.open("result/summary_result_new.jsonl", "a") as f:
    #     f.write(state)
    final_score= weighting_scheme(state)
    print('final_score',final_score)

    # Build per-sentence records alongside existing arrays (additive, no arrays changed)
    records = build_sentence_records(
        state,
        sentence_scores=sentence_scores,
        failing_sentence_indices=failing_sentence_indices,
        refine_counts=refine_counts,
        threshold=sentence_threshold,
        max_refines=MAX_SENTENCE_REFINES,
    )
    records_consistency = validate_sentence_records_consistency(
        sentence_scores=sentence_scores,
        refine_counts=refine_counts,
        records=records,
    )

    return {
        "score": weighting_scheme(state),
        "summary": summary,
        "sentence_scores": sentence_scores,
        "failing_sentence_indices": failing_sentence_indices,
        "instance_refine_counts": refine_counts,
        "sentence_records": records,
        "records_consistency": records_consistency,
    }


# index-aligned per-instance arrays that must be filtered together when dropping sentences
_PER_INSTANCE_KEYS = (
    "data_generation_result",
    "sentence_records",
    "fluency_results_per_instances",
    "naturalness_results_per_instances",
    "cs_ratio_results_per_instances",
    "social_cultural_results_per_instances",
    "task_validation_results_per_instances",
    "instance_refine_counts",
    "sentence_scores",
)


def _filter_state_to_task_passed(state: AgentRunningState) -> AgentRunningState:
    """Return a copy of `state` keeping only sentences whose FINAL task verdict passed.

    A sentence is DROPPED only when its verdict is explicitly False (None/unknown is
    kept, so disabling the validator never empties the corpus). All index-aligned
    per-instance arrays are filtered by the same kept indices; sentence_records are
    re-indexed and failing_sentence_indices recomputed. Safe no-op if there is nothing
    to drop or sentence_records is absent. Used only when TASK_AWARE_ACCEPT is enabled.
    """
    records = state.get("sentence_records")
    if not isinstance(records, list) or not records:
        return state
    n = len(records)
    kept = [i for i, rec in enumerate(records)
            if not (isinstance(rec, dict) and rec.get("task_passed") is False)]
    if len(kept) == n:
        return state  # nothing to drop
    out = dict(state)
    for key in _PER_INSTANCE_KEYS:
        val = state.get(key)
        if isinstance(val, list) and len(val) == n:
            out[key] = [val[i] for i in kept]
    new_records = out.get("sentence_records")
    if isinstance(new_records, list):
        for new_i, rec in enumerate(new_records):
            if isinstance(rec, dict):
                rec = dict(rec)
                rec["index"] = new_i
                new_records[new_i] = rec
        out["failing_sentence_indices"] = [
            j for j, rec in enumerate(new_records)
            if isinstance(rec, dict) and isinstance(rec.get("weighted_score"), (int, float))
            and float(rec["weighted_score"]) < 8.0
        ]
    return out


def AcceptanceAgent(state: AgentRunningState):
    state.pop("news_article", None)
    state.pop("news_hash", None)
    state.pop("news_dict", None)
    # OPT-IN task-aware acceptance: drop sentences that still fail the task (default OFF).
    out = _filter_state_to_task_passed(state) if TASK_AWARE_ACCEPT else state
    language = out.get("first_language", "Unknown")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with jsonlines.open(
        f"{OUTPUT_DIR}/{language}.jsonl",
        "a",
    ) as f:
        f.write(out)
    # Return the (possibly filtered) state so consumers that read the RETURNED graph
    # state — not just the JSONL file — also see task-aware acceptance. With the flag
    # OFF, `out is state`, so this merges identical values (no behaviour change).
    return out


def _candidate_passes_task(state: dict, candidate: str, val_prompt) -> bool:
    """Re-validate a single refined candidate against the task (guardrail helper).
    Mirrors the historical semantics: a non-dict validator response or a missing
    'passed' key counts as a pass (never reject on validator malfunction)."""
    candidate_state = dict(state)
    candidate_state["data_generation_result"] = [candidate]
    result = _invoke_task_validation_with_retry(candidate_state, val_prompt)
    return not (isinstance(result, dict) and not result.get("passed", True))


def _rescore_single_sentence(text: str, state: dict) -> float:
    """Re-score one sentence by reusing the existing quality agents.
    Returns weighted score using the same formula as compute_sentence_weighted_scores().
    """
    candidate_state = dict(state)
    candidate_state["data_generation_result"] = [text]

    candidate_state.update(RunFluencyAgent(candidate_state))
    candidate_state.update(RunNaturalnessAgent(candidate_state))
    candidate_state.update(RunCSRatioAgent(candidate_state))
    candidate_state.update(RunSocialCulturalAgent(candidate_state))

    scores = compute_sentence_weighted_scores(candidate_state)
    return scores[0] if scores else 0.0


def RunRefinerAgent(state: AgentRunningState):
    texts = state.get("data_generation_result", [])
    if not isinstance(texts, list) or not texts:
        return {}

    records = state.get("sentence_records", [])
    if not isinstance(records, list) or len(records) != len(texts):
        return {"instance_refine_counts": state.get("instance_refine_counts", [])}

    failing_indices = []
    failure_reasons = {}  # index -> "task_fail" or "quality_fail"
    refine_counts = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            refine_counts.append(0)
            continue
        rc = int(rec.get("refine_count", 0) or 0)
        refine_counts.append(rc)

        # Determine why this sentence is failing
        task_val = rec.get("task_validation") if isinstance(rec.get("task_validation"), dict) else {}
        if not task_val:
            per_instance = state.get("task_validation_results_per_instances", [])
            if isinstance(per_instance, list) and i < len(per_instance) and isinstance(per_instance[i], dict):
                task_val = per_instance[i]
        task_passed = bool(task_val.get("passed", True))  # default True if unknown

        weighted_score = rec.get("weighted_score")
        quality_low = isinstance(weighted_score, (int, float)) and float(weighted_score) < 8.0

        if not task_passed:
            failing_indices.append(i)
            failure_reasons[i] = "task_fail"
        elif quality_low:
            failing_indices.append(i)
            failure_reasons[i] = "quality_fail"

    eligible_indices = [
        idx
        for idx in failing_indices
        if isinstance(idx, int)
        and 0 <= idx < len(texts)
        and int(refine_counts[idx]) < MAX_SENTENCE_REFINES
    ]

    if not eligible_indices:
        return {"instance_refine_counts": refine_counts}

    llm_refiner = ChatOpenAI(model=MODEL, temperature=0.1, base_url=API_BASE, api_key=API_KEY).with_structured_output(GenerationResponse)
    task_type = state.get("task", "")

    updated_texts = list(texts)
    for index in eligible_indices:
        rec = records[index] if isinstance(records[index], dict) else {}
        score_at_index = rec.get("weighted_score")
        flu_inst = rec.get("fluency") if isinstance(rec.get("fluency"), dict) else {}
        nat_inst = rec.get("naturalness") if isinstance(rec.get("naturalness"), dict) else {}
        cs_inst = rec.get("cs_ratio") if isinstance(rec.get("cs_ratio"), dict) else {}
        soc_inst = rec.get("socio_cultural") if isinstance(rec.get("socio_cultural"), dict) else {}

        task_val_inst = rec.get("task_validation") if isinstance(rec.get("task_validation"), dict) else {}
        if not task_val_inst:
            per_instance = state.get("task_validation_results_per_instances", [])
            if isinstance(per_instance, list) and index < len(per_instance) and isinstance(per_instance[index], dict):
                task_val_inst = per_instance[index]

        single_state = dict(state)
        single_state["data_generation_result"] = [texts[index]]
        single_state["sentence_index"] = index
        single_state["sentence_score"] = score_at_index
        single_state["fluency_result"] = flu_inst
        single_state["naturalness_result"] = nat_inst
        single_state["cs_ratio_result"] = cs_inst
        single_state["social_cultural_result"] = soc_inst
        if task_val_inst:
            single_state["task_validation_result"] = task_val_inst
        failure_reason = failure_reasons.get(index, "quality_fail")
        single_state["failure_reason"] = failure_reason

        # Select prompt based on failure reason and task type
        if failure_reason == "task_fail":
            if task_type == "topic":
                single_state["topic"] = state.get("topic", "")
                refiner = REFINER_TASK_TOPIC_PROMPT | llm_refiner
            elif task_type == "sentiment":
                single_state["label"] = state.get("label", "")
                refiner = REFINER_TASK_SENTIMENT_PROMPT | llm_refiner
            elif task_type == "ner":
                constraints = state.get("task_constraints", {})
                single_state["entity_types"] = ", ".join(constraints.get("entity_types", []))
                refiner = REFINER_TASK_NER_PROMPT | llm_refiner
            else:
                refiner = REFINER_PROMPT | llm_refiner
        else:
            refiner = REFINER_PROMPT | llm_refiner

        single_state["sentence"] = texts[index]
        single_state["task_validation_feedback"] = task_val_inst.get("feedback", "") if task_val_inst else ""
        single_state["summary"] = (
            f"Sentence index: {index}\n"
            f"Original sentence: {texts[index]}\n"
            f"Failure reason: {failure_reason}\n"
            f"Sentence score: {score_at_index}\n"
            f"Fluency: {flu_inst}\n"
            f"Naturalness: {nat_inst}\n"
            f"CS ratio: {cs_inst}\n"
            f"Social cultural: {soc_inst}\n"
            f"Task validation: {task_val_inst}"
        )

        response = refiner.invoke(single_state)
        refined_instances = response.get("instances", []) if isinstance(response, dict) else []
        applied = False
        if isinstance(refined_instances, list) and refined_instances:
            candidate = refined_instances[0]
            if isinstance(candidate, str) and candidate.strip():
                candidate = candidate.strip()

                # --- Rewrite Guardrail (keep candidate vs roll back to original) ---
                # Decides which TEXT survives; it does NOT decide corpus acceptance
                # (that is meet_criteria / AcceptanceAgent). Asymmetric by design:
                #   quality_fail: keep candidate only if task still passes AND quality
                #                 did not regress;
                #   task_fail:    keep candidate only if task now passes (quality
                #                 regression tolerated — the goal is the task repair).
                val_prompt = {
                    "topic": TASK_VALIDATION_TOPIC_PROMPT,
                    "sentiment": TASK_VALIDATION_SENTIMENT_PROMPT,
                    "ner": TASK_VALIDATION_NER_PROMPT,
                }.get(task_type)
                accept = True
                if val_prompt is not None:
                    candidate_task_ok = _candidate_passes_task(state, candidate, val_prompt)
                    if not candidate_task_ok:
                        accept = False  # Rollback: candidate fails (or broke) the task
                    elif failure_reason == "quality_fail":
                        # Task intact — additionally require no quality regression
                        before_quality_score = float(rec.get("weighted_score") or 0.0)
                        if _rescore_single_sentence(candidate, state) < before_quality_score:
                            accept = False  # Rollback: quality regressed after refine

                if accept:
                    updated_texts[index] = candidate
                    applied = True
        # Count every refinement ATTEMPT against the per-sentence budget — whether or not
        # the guardrail accepted the new text. Otherwise a sentence that cannot be safely
        # improved (e.g. fixing wording would break required NER entities) keeps its
        # refine_count at 0, so meet_criteria re-routes to the refiner forever (infinite loop).
        # After the budget is spent the sentence is accepted as 'budget_exhausted'.
        refine_counts[index] = int(refine_counts[index]) + 1

    return {
        "data_generation_result": updated_texts,
        "refine_count": 1,
        "instance_refine_counts": refine_counts,
    }

def RunMCPAgent(state: AgentRunningState) -> Dict[str, Any]:
    """
    Iterate through all MCP tools in the registry, execute them in order, and merge the results.
    The execution result -> state["mcp_result"], used by the subsequent nodes.
    """
    result: Dict[str, Any] = {}
    for tool_name, tool in get_all_tools().items():
        try:
            result.update(tool.run(state))
        except Exception as e:
            # Ensure that a tool failure does not affect the subsequent nodes
            result[tool_name] = f"ERROR: {e}"
    return {"mcp_result": result}
