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
from utils import weighting_scheme, compute_sentence_weighted_scores, build_sentence_records
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
MAX_SENTENCE_REFINES = int(os.getenv("MAX_SENTENCE_REFINES", "1"))

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


def RunNERDataGenerationAgent(state: AgentRunningState):
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
        return _invoke_task_validation_with_retry(state, validator_prompt)

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
            "passed": False,
            "confidence": 0.0,
            "notes": "Validator returned empty per-instance responses",
            "predicted_label": None,
            "errors": ["empty_per_instance_validator_response"],
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
        "passed": passed,
        "confidence": confidence,
        "notes": notes,
        "predicted_label": aggregate_predicted,
        "errors": all_errors,
        "per_instance_results": per_instance_results,
    }


def RunTopicTaskValidatorAgent(state: AgentRunningState):
    response = _validate_per_instance_with_retry(state, TASK_VALIDATION_TOPIC_PROMPT)
    return {"task_validation_result": response}


def RunSentimentTaskValidatorAgent(state: AgentRunningState):
    response = _validate_per_instance_with_retry(state, TASK_VALIDATION_SENTIMENT_PROMPT)
    return {"task_validation_result": response}


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
            r"\b(?:company|organization|institution|bank|جامعة|شركة|بنك)\s+([A-Z][A-Za-z0-9&\.-]*(?:\s+[A-Z][A-Za-z0-9&\.-]*)?)\b",
            ascii_text,
            flags=re.IGNORECASE,
        )
    )
    org_entities.update(
        re.findall(
            r"\b(?:program|app|application|platform|tool|service|product|برنامج|تطبيق|منصة|خدمة|منتج)\s+([A-Z][A-Za-z0-9&\.-]*(?:\s+[A-Z][A-Za-z0-9&\.-]*)?)\b",
            ascii_text,
            flags=re.IGNORECASE,
        )
    )
    org_entities.update(
        re.findall(
            r"\b(?:from|by|via|using|use|about|with|من|عن|باستخدام)\s+([A-Z][A-Za-z0-9&\.-]*(?:\s+[A-Z][A-Za-z0-9&\.-]*)?)\b",
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
        re.findall(r"\bفي\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text)
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
    # deterministic = _deterministic_ner_english_policy(state)
    llm_notes = llm_response.get("notes", "") if isinstance(llm_response, dict) else ""

    final_result = {
        "passed": bool(llm_response.get("passed", False)) if isinstance(llm_response, dict) else False,
        "confidence": float(llm_response.get("confidence", 0.0)) if isinstance(llm_response, dict) else 0.0,
        "notes": llm_notes,
        "deterministic_notes": "",
        "llm_notes": llm_notes,
        "predicted_label": llm_response.get("predicted_label") if isinstance(llm_response, dict) else None,
        "errors": llm_response.get("errors", []) if isinstance(llm_response, dict) else ["empty_validator_response"],
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
    return {"task_validation_result": final_result}


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
        print("WARNING: DataGenerationAgent failed to generate instances after retries")
        return {
            "data_generation_result": [],
        }

    # # 🔹 NEW PART STARTS HERE 🔹
    # per_instance_stats = [compute_true_cs_stats(x) for x in instances]

    # # instances is usually a list of strings
    # if isinstance(instances, list):
    #     text_for_stats = " ".join(instances)
    # else:
    #     text_for_stats = str(instances)
    # print('text for stats',text_for_stats)
    # cs_stats = compute_true_cs_stats(text_for_stats)
    # print("CS STATS:", cs_stats)

    # # 🔹 NEW PART ENDS HERE 🔹

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

    FluencyAgent = FLUENCY_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, base_url=API_BASE, api_key=API_KEY
    )
    response = FluencyAgent.invoke(
        {"sentences_for_batch": _build_sentences_for_batch(texts)}
    )
    batch_results = _extract_json_array(response)

    results = []
    for idx in range(len(texts)):
        item = batch_results[idx] if idx < len(batch_results) else {}
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

    NaturalnessAgent = NATURALNESS_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, base_url=API_BASE, api_key=API_KEY
    )
    response = NaturalnessAgent.invoke(
        {"sentences_for_batch": _build_sentences_for_batch(texts)}
    )
    batch_results = _extract_json_array(response)

    results = []
    for idx in range(len(texts)):
        item = batch_results[idx] if idx < len(batch_results) else {}
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
        target_ar = float(cs_ratio.replace("%","")) 
        target_en = 100 - target_ar
        return round(target_en, 4), round(target_ar, 4)
    except (ValueError, AttributeError) as e:
        print(f"ERROR parsing cs_ratio '{cs_ratio}': {e}")
        return 0.0, 0.0
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


def _safe_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def RunCSRatioAgent(state: AgentRunningState):
    print("CSRatio received keys:", state.keys())
    results = []
    texts = state.get("data_generation_result", [])
    
    if not texts:
        print("WARNING: No texts to evaluate in RunCSRatioAgent")
        return {"cs_ratio_results_per_instances": []}
    
    per_scenario_stats = [compute_true_cs_stats(x) for x in texts]
    print("stats_list length:", len(per_scenario_stats))
    print("stats_list:", per_scenario_stats)

    target_en, target_ar = parse_target_ratios(state.get("cs_ratio"))
    print('target', target_en, target_ar)
    
    # Prepare batch data: combine sentences with their stats
    sentences_with_stats = []
    for i, (sent, stats) in enumerate(zip(texts, per_scenario_stats)):
        sentences_with_stats.append(
            f"Sentence {i+1}: {sent}\n"
            f"  - Arabic ratio: {stats['cs_ar_ratio']:.2f}%\n"
            f"  - English ratio: {stats['cs_en_ratio']:.2f}%\n"
            f"  - Arabic tokens: {stats['cs_ar_count']}\n"
            f"  - English tokens: {stats['cs_en_count']}\n"
            f"  - Is code-switched: {stats['is_code_switched']}\n"
        )
    
    sentences_with_stats_str = "\n".join(sentences_with_stats)
    
    # Batch call: all sentences at once
    local_state = {
        "cs_ratio": state.get("cs_ratio"),
        "target_en_ratio": target_en,
        "target_ar_ratio": target_ar,
        "sentences_with_stats": sentences_with_stats_str,
    }
    
    CSRatioAgent = CS_RATIO_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, base_url=API_BASE, api_key=API_KEY
    )
    
    response = CSRatioAgent.invoke(local_state)
    
    # Parse response - expecting JSON array
    import json
    try:
        content = response.content if hasattr(response, 'content') else str(response)
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            batch_results = json.loads(json_str)
        else:
            batch_results = json.loads(content)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"ERROR parsing batch response: {e}")
        batch_results = []
    
    # Ensure we have results for each sentence
    if isinstance(batch_results, list):
        results = batch_results[:len(texts)]
    else:
        results = [batch_results] if batch_results else []
    
    # Fill in any missing results from stats
    while len(results) < len(texts):
        stats = per_scenario_stats[len(results)]
        results.append({
            "ratio_score": 0 if not stats['is_code_switched'] else 5,
            "computed_ratio": f"{stats['cs_ar_ratio']:.2f}% : {stats['cs_en_ratio']:.2f}%",
            "notes": "monolingual" if not stats['is_code_switched'] else "code-switched"
        })
    
    return {"cs_ratio_results_per_instances": results}










    # stats_list = state.get("cs_stats_per_instance", [])
    # print("stats_list length:", len(stats_list))


    # ########
    # target_en, target_ar = parse_target_ratios(state.get("cs_ratio"))


    # for sent, stats in zip(texts, stats_list):
    #     local_state = {
    #         "cs_ratio": state.get("cs_ratio"),
    #         "target_en_ratio": target_en,
    #         "target_ar_ratio": target_ar,
    #         "data_generation_result": sent,  # pass single sentence
    #         **stats,                         # deterministic ratios for this sentence
    #     }
    #     response_1=CSRatioAgent.invoke(local_state)
    #     if not local_state.get("is_code_switched"):
    #         set_field(response_1, "ratio_score", 0)
    #         set_field(response_1, "notes", "monolingual")
    #     if state.get("cs_ratio") == "70%":
    #         assert abs(local_state["target_en_ratio"] - 0.30) < 1e-6
    #     results.append(response_1)

    return {
      "cs_ratio_results_per_instances":results
      # "cs_ratio_results_per_instance": results,
      # "cs_ratio_result": results[0] if results else None
    }

    ################################################ grbiha
    # # HARD RULE: if monolingual -> cap score
    # if state.get("is_code_switched") is False:
    #     response.ratio_score = min(response.ratio_score, 2.0)
    #     response.notes = (response.notes or "") + " | Hard rule: monolingual output."

    # return {"cs_ratio_result": response}


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

    SocialCulturalAgent = SOCIAL_CULTURAL_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, base_url=API_BASE, api_key=API_KEY
    )
    response = SocialCulturalAgent.invoke(
        {"sentences_for_batch": _build_sentences_for_batch(texts)}
    )
    batch_results = _extract_json_array(response)

    results = []
    for idx in range(len(texts)):
        item = batch_results[idx] if idx < len(batch_results) else {}
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

    return {
        "score": weighting_scheme(state),
        "summary": summary,
        "sentence_scores": sentence_scores,
        "failing_sentence_indices": failing_sentence_indices,
        "instance_refine_counts": refine_counts,
        "sentence_records": records,
    }


def AcceptanceAgent(state: AgentRunningState):
    state.pop("news_article", None)
    state.pop("news_hash", None)
    state.pop("news_dict", None)
    language = state.get("first_language", "Unknown")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with jsonlines.open(
        f"{OUTPUT_DIR}/{language}.jsonl",
        "a",
    ) as f:
        f.write(state)
    return


def RunRefinerAgent(state: AgentRunningState):
    texts = state.get("data_generation_result", [])
    if not isinstance(texts, list) or not texts:
        return {"refine_count": 1}

    failing_indices = state.get("failing_sentence_indices", [])
    if not isinstance(failing_indices, list):
        failing_indices = []

    refine_counts = state.get("instance_refine_counts", [])
    if not isinstance(refine_counts, list):
        refine_counts = []
    if len(refine_counts) < len(texts):
        refine_counts = refine_counts + [0] * (len(texts) - len(refine_counts))

    eligible_indices = [
        idx
        for idx in failing_indices
        if isinstance(idx, int)
        and 0 <= idx < len(texts)
        and int(refine_counts[idx]) < MAX_SENTENCE_REFINES
    ]

    if not eligible_indices:
        return {"refine_count": 1, "instance_refine_counts": refine_counts}

    refiner = REFINER_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, base_url=API_BASE, api_key=API_KEY
    ).with_structured_output(GenerationResponse)

    updated_texts = list(texts)
    for index in eligible_indices:
        sentence_scores = state.get("sentence_scores", [])
        score_at_index = (
            sentence_scores[index]
            if isinstance(sentence_scores, list) and index < len(sentence_scores)
            else None
        )

        flu_all = state.get("fluency_results_per_instances", [])
        nat_all = state.get("naturalness_results_per_instances", [])
        cs_all = state.get("cs_ratio_results_per_instances", [])
        soc_all = state.get("social_cultural_results_per_instances", [])

        flu_inst = flu_all[index] if isinstance(flu_all, list) and index < len(flu_all) and isinstance(flu_all[index], dict) else {}
        nat_inst = nat_all[index] if isinstance(nat_all, list) and index < len(nat_all) and isinstance(nat_all[index], dict) else {}
        cs_inst = cs_all[index] if isinstance(cs_all, list) and index < len(cs_all) and isinstance(cs_all[index], dict) else {}
        soc_inst = soc_all[index] if isinstance(soc_all, list) and index < len(soc_all) and isinstance(soc_all[index], dict) else {}

        task_val_inst = {}
        tvr = state.get("task_validation_result", {})
        if isinstance(tvr, dict):
            per_instance = tvr.get("per_instance_results", [])
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
        single_state["summary"] = (
            f"Sentence index: {index}\n"
            f"Original sentence: {texts[index]}\n"
            f"Sentence score: {score_at_index}\n"
            f"Fluency: {flu_inst}\n"
            f"Naturalness: {nat_inst}\n"
            f"CS ratio: {cs_inst}\n"
            f"Social cultural: {soc_inst}\n"
            f"Task validation: {task_val_inst}"
        )

        response = refiner.invoke(single_state)
        refined_instances = response.get("instances", []) if isinstance(response, dict) else []
        if isinstance(refined_instances, list) and refined_instances:
            candidate = refined_instances[0]
            if isinstance(candidate, str) and candidate.strip():
                updated_texts[index] = candidate.strip()
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