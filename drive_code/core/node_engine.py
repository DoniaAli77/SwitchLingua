import os
from utils import compute_true_cs_stats

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
from typing import Dict, Any
# from google.colab import userdata

dotenv.load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE = os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE")
MODEL = "gpt-4o-mini"
OUTPUT_DIR = "output"

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

    DataGenerationAgent = DATA_GENERATION_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.7, base_url=API_BASE, api_key=API_KEY
    ).with_structured_output(GenerationResponse)

    response = DataGenerationAgent.invoke(state)

    retry = 4
    if not response.get("instances"):
        while retry > 0:
            response = DataGenerationAgent.invoke(state)
            if response.get("instances"):
                break
            retry -= 1

    instances = response.get("instances", [])
    if not instances:
        print("WARNING: DataGenerationAgent failed to generate instances after retries")
        return {"data_generation_result": []}

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
    CSRatio Result: {state["cs_ratio_results_per_instances"]}
    Social Cultural Result: {state["social_cultural_result"]}
    """
    state["summary"] = summary
    # print(summary)
    # with jsonlines.open("result/summary_result_new.jsonl", "a") as f:
    #     f.write(state)
    final_score= weighting_scheme(state)
    print('final_score',final_score)
    return {"score": weighting_scheme(state), "summary": summary}


def AcceptanceAgent(state: AgentRunningState):
    state.pop("news_article", None)
    state.pop("news_hash", None)
    state.pop("news_dict", None)
    language = state.get("first_language", "Unknown")
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