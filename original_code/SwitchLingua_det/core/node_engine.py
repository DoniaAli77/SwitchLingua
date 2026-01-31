import os
import traceback
import dotenv
import random
import jsonlines
import unicodedata
import re
try:
    from langdetect import detect, LangDetectException
    _LANGDETECT_AVAILABLE = True
except Exception:
    _LANGDETECT_AVAILABLE = False
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from prompt import (
    DATA_GENERATION_PROMPT,
    FLUENCY_PROMPT,
    NATURALNESS_PROMPT,
    CS_RATIO_SCORE_PROMPT,
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
)
from utils import weighting_scheme
from copy import deepcopy

from mcp_tools import get_all_tools


dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE = os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE")
MODEL = "gpt-4o-mini"
OUTPUT_DIR = "output"
print("OPENAI_API_KEY set:", bool(API_KEY))

def _language_to_group(language: str) -> str:
    lang = (language or "").strip().lower()
    if lang in {"arabic"}:
        return "arabic"
    if lang in {"chinese", "mandarin", "cantonese"}:
        return "han"
    if lang in {"japanese"}:
        return "japanese"
    if lang in {"korean"}:
        return "hangul"
    if lang in {"russian", "ukrainian", "bulgarian", "serbian"}:
        return "cyrillic"
    if lang in {"greek"}:
        return "greek"
    if lang in {"hebrew"}:
        return "hebrew"
    if lang in {"hindi"}:
        return "devanagari"
    if lang in {"thai"}:
        return "thai"
    if lang in {
        "english",
        "spanish",
        "french",
        "german",
        "italian",
        "portuguese",
        "catalan",
        "turkish",
        "vietnamese",
        "indonesian",
        "malay",
    }:
        return "latin"
    return "unknown"

def _char_in_group(ch: str, group: str) -> bool:
    code = ord(ch)
    if group == "latin":
        name = unicodedata.name(ch, "")
        return "LATIN" in name
    if group == "arabic":
        return (
            0x0600 <= code <= 0x06FF
            or 0x0750 <= code <= 0x077F
            or 0x08A0 <= code <= 0x08FF
            or 0xFB50 <= code <= 0xFDFF
            or 0xFE70 <= code <= 0xFEFF
        )
    if group == "han":
        return (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0xF900 <= code <= 0xFAFF
        )
    if group == "japanese":
        return (
            0x3040 <= code <= 0x309F
            or 0x30A0 <= code <= 0x30FF
            or 0x31F0 <= code <= 0x31FF
            or 0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0xF900 <= code <= 0xFAFF
        )
    if group == "hangul":
        return (
            0xAC00 <= code <= 0xD7AF
            or 0x1100 <= code <= 0x11FF
            or 0x3130 <= code <= 0x318F
        )
    if group == "cyrillic":
        return (
            0x0400 <= code <= 0x04FF
            or 0x0500 <= code <= 0x052F
            or 0x2DE0 <= code <= 0x2DFF
            or 0xA640 <= code <= 0xA69F
        )
    if group == "greek":
        return 0x0370 <= code <= 0x03FF or 0x1F00 <= code <= 0x1FFF
    if group == "hebrew":
        return 0x0590 <= code <= 0x05FF
    if group == "devanagari":
        return 0x0900 <= code <= 0x097F
    if group == "thai":
        return 0x0E00 <= code <= 0x0E7F
    return False

def _score_ratio(target_pct: float, actual_pct: float) -> float:
    diff = abs(actual_pct - target_pct)
    if diff <= 5:
        return 10.0
    if diff <= 10:
        return 8.0
    if diff <= 15:
        return 6.0
    if diff <= 20:
        return 4.0
    if diff <= 25:
        return 2.0
    return 0.0

def _deterministic_cs_ratio(state: AgentRunningState) -> CSRatioResponse:
    matrix_lang = state.get("first_language", "")
    embedded_lang = state.get("second_language", "")
    matrix_group = _language_to_group(matrix_lang)
    embedded_group = _language_to_group(embedded_lang)
    target_raw = state.get("cs_ratio", "0%").strip().replace("%", "")
    try:
        target_pct = float(target_raw)
    except ValueError:
        target_pct = 0.0

    if matrix_group == "unknown" or embedded_group == "unknown":
        return {
            "computed_ratio": "N/A",
            "ratio_score": 0.0,
            "notes": "Unknown script mapping for one or both languages; deterministic ratio not computed.",
        }

    if matrix_group == embedded_group:
        return {
            "computed_ratio": "N/A",
            "ratio_score": 0.0,
            "notes": "Both languages share the same script; deterministic script-based ratio not possible.",
        }

    data = state.get("data_generation_result") or []
    if isinstance(data, list):
        sentences = [str(x) for x in data]
    else:
        sentences = [str(data)]

    matrix_count = 0
    embedded_count = 0
    word_re = re.compile(r"[\w']+", re.UNICODE)
    for sentence in sentences:
        for word in word_re.findall(sentence):
            w = word.strip()
            if not w:
                continue
            detected = None
            if _LANGDETECT_AVAILABLE:
                try:
                    detected = detect(w)
                except LangDetectException:
                    detected = None

            if detected:
                det = detected.lower()
                # Prefer explicit language match when available
                if det.startswith((matrix_lang or "").lower()[:2]):
                    matrix_count += 1
                    continue
                if det.startswith((embedded_lang or "").lower()[:2]):
                    embedded_count += 1
                    continue

            # Fallback: script majority per word
            m_chars = 0
            e_chars = 0
            for ch in w:
                if _char_in_group(ch, matrix_group):
                    m_chars += 1
                elif _char_in_group(ch, embedded_group):
                    e_chars += 1
            if m_chars > e_chars:
                matrix_count += 1
            elif e_chars > m_chars:
                embedded_count += 1

    total = matrix_count + embedded_count
    if total == 0:
        return {
            "computed_ratio": "N/A",
            "ratio_score": 0.0,
            "notes": "No identifiable script characters found for either language.",
        }

    matrix_pct = round((matrix_count / total) * 100)
    embedded_pct = 100 - matrix_pct
    score = _score_ratio(target_pct, matrix_pct)
    return {
        "computed_ratio": f"{matrix_pct}% : {embedded_pct}%",
        "ratio_score": score,
        "notes": (
            f"Deterministic script ratio using {matrix_lang}/{embedded_lang} "
            f"({matrix_group}/{embedded_group})."
        ),
    }

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


def RunDataGenerationAgent(state: AgentRunningState):
    if state.get("topic") not in state["news_dict"]:
        state["news_article"] = ""
    else:
        if state.get("topic") in state["news_hash"]:
            state["news_article"] = random.choice(state["news_dict"][state["topic"]])
        else:
            state["news_article"] = random.choice(state["news_dict"][state["topic"]])
            state["news_hash"].add(state["topic"])
    DataGenerationAgent = DATA_GENERATION_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.7, base_url=API_BASE, api_key=API_KEY
    ).with_structured_output(GenerationResponse)
    try:
        response = DataGenerationAgent.invoke(state)
    except Exception as e:
        print("DataGenerationAgent error:", repr(e))
        traceback.print_exc()
        raise
    retry = 4
    if not response.get("instances"):
        while retry > 0:
            response = DataGenerationAgent.invoke(state)
            if response.get("instances"):
                break
            retry -= 1
    return {"data_generation_result": response["instances"]}


def RunFluencyAgent(state: AgentRunningState):
    FluencyAgent = FLUENCY_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, base_url=API_BASE, api_key=API_KEY
    ).with_structured_output(FluencyResponse)
    response = FluencyAgent.invoke(state)

    return {"fluency_result": response}


def RunNaturalnessAgent(state: AgentRunningState):
    NaturalnessAgent = NATURALNESS_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, base_url=API_BASE, api_key=API_KEY
    ).with_structured_output(NaturalnessResponse)
    response = NaturalnessAgent.invoke(state)

    return {"naturalness_result": response}


def RunCSRatioAgent(state: AgentRunningState):
    det = _deterministic_cs_ratio(state)
    if det.get("computed_ratio") in {None, "N/A"}:
        return {"cs_ratio_result": det}

    ScoringAgent = CS_RATIO_SCORE_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, base_url=API_BASE, api_key=API_KEY
    ).with_structured_output(CSRatioResponse)
    payload = dict(state)
    payload["computed_ratio"] = det["computed_ratio"]
    response = ScoringAgent.invoke(payload)
    response["computed_ratio"] = det["computed_ratio"]
    return {"cs_ratio_result": response}


def RunSocialCulturalAgent(state: AgentRunningState):
    SocialCulturalAgent = SOCIAL_CULTURAL_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, base_url=API_BASE, api_key=API_KEY
    ).with_structured_output(SocialCulturalResponse)
    response = SocialCulturalAgent.invoke(state)
    return {"social_cultural_result": response}


def SummarizeResult(state: AgentRunningState):
    summary = f"""
    data_generation_result: {state["data_generation_result"]}
    Fluency Result: {state["fluency_result"]}
    Naturalness Result: {state["naturalness_result"]}
    CSRatio Result: {state["cs_ratio_result"]}
    Social Cultural Result: {state["social_cultural_result"]}
    """
    state["summary"] = summary
    # print(summary)
    # with jsonlines.open("result/summary_result_new.jsonl", "a") as f:
    #     f.write(state)

    return {"score": weighting_scheme(state), "summary": summary}


def AcceptanceAgent(state: AgentRunningState):
    del state["news_article"]
    del state["news_hash"]
    del state["news_dict"]
    language = state["first_language"]
    with jsonlines.open(
        f"{OUTPUT_DIR}/{language}.jsonl",
        "a",
    ) as f:
        f.write(state)
    return


def RunRefinerAgent(state: AgentRunningState):

    RefinerAgent = REFINER_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, base_url=API_BASE, api_key=API_KEY
    ).with_structured_output(GenerationResponse)
    response = RefinerAgent.invoke(state)

    return {"refiner_result": response, "refine_count": 3}

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
