import os
import dotenv
import random
import jsonlines
from langchain_openai import ChatOpenAI
from prompt import (
    DATA_GENERATION_PROMPT,
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
)
from utils import weighting_scheme
from copy import deepcopy

from mcp_tools import get_all_tools
from cs_ratio_calculator import compute_cs_ratio, calculate_ratio_score
from typing import Dict, Any


dotenv.load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE = os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE")
MODEL = "gpt-4o-mini"
OUTPUT_DIR = "output"

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
        model=MODEL, temperature=0.7, base_url=API_BASE
    ).with_structured_output(GenerationResponse)
    response = DataGenerationAgent.invoke(state)
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
        model=MODEL, temperature=0.1, base_url=API_BASE
    ).with_structured_output(NaturalnessResponse)
    response = NaturalnessAgent.invoke(state)

    return {"naturalness_result": response}


def RunCSRatioAgent(state: AgentRunningState):
    # HYBRID APPROACH: Deterministic calculation + LLM interpretation
    
    # Step 1: Deterministic calculation (accurate word counting)
    ratio_data = compute_cs_ratio(
        state["data_generation_result"],
        state["first_language"],
        state["second_language"]
    )
    
    # Step 2: Calculate score based on target vs actual
    ratio_score = calculate_ratio_score(
        ratio_data["lang2_percent"],
        state["cs_ratio"]
    )
    
    # Step 3: Use LLM for qualitative feedback (optional enhancement)
    # This adds interpretation without relying on it for accuracy
    try:
        CSRatioAgent = CS_RATIO_PROMPT | ChatOpenAI(
            model=MODEL, temperature=0.1, base_url=API_BASE, api_key=API_KEY
        ).with_structured_output(CSRatioResponse)
        
        # Compute explicit target percents and pass the accurate data to LLM
        try:
            target_pct = float(str(state.get("cs_ratio", "0")).rstrip("%"))
        except Exception:
            target_pct = 0.0

        target_second_percent = f"{target_pct:.1f}%"  # embedded / second language
        target_first_percent = f"{(100.0 - target_pct):.1f}%"  # matrix / first language

        llm_state = state.copy()
        llm_state["computed_ratio"] = ratio_data["computed_ratio"]
        llm_state["actual_percent"] = f"{ratio_data['lang2_percent']:.1f}%"
        llm_state["target_second_percent"] = target_second_percent
        llm_state["target_first_percent"] = target_first_percent
        
        llm_response = CSRatioAgent.invoke(llm_state)
        
        # Use LLM's notes/analysis but override the score with our accurate calculation
        response = {
            "ratio_score": ratio_score,  # ← Use deterministic score
            "computed_ratio": ratio_data["computed_ratio"],  # ← Use accurate ratio
            "notes": llm_response.get("notes", ratio_data["details"])  # ← Use LLM notes if available
        }
    except Exception as e:
        # Fallback if LLM fails: use pure deterministic results
        response = {
            "ratio_score": ratio_score,
            "computed_ratio": ratio_data["computed_ratio"],
            "notes": f"{ratio_data['details']} Target: {state['cs_ratio']}"
        }
    
    return {"cs_ratio_result": response}


def RunSocialCulturalAgent(state: AgentRunningState):
    SocialCulturalAgent = SOCIAL_CULTURAL_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, base_url=API_BASE
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
        model=MODEL, temperature=0.1, base_url=API_BASE
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