import langchain
import os

# --- Compatibility patch for older LangChain assumptions ---
if not hasattr(langchain, "debug"):
    langchain.debug = False

if not hasattr(langchain, "verbose"):
    langchain.verbose = False

if not hasattr(langchain, "llm_cache"):
    langchain.llm_cache = None


import asyncio
from langgraph.graph import StateGraph, START, END
from loguru import logger
from utils import load_config, generate_scenarios
from node_engine import (
    RunMCPAgent,
    RunDataGenerationAgent,
    RunTaskValidatorAgent,
    SummarizeResult,
    RunFluencyAgent,
    RunNaturalnessAgent,
    RunCSRatioAgent,
    RunSocialCulturalAgent,
    RunRefinerAgent,
    AcceptanceAgent,
)
from node_models import AgentRunningState
import random
from tqdm import tqdm
import jsonlines as jsl
from datetime import datetime

logger.add(f"logs/code_switching_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

MAX_REFINER_ITERATIONS = 1
ENABLE_TASK_VALIDATOR = os.getenv("ENABLE_TASK_VALIDATOR", "1").strip() == "1"
SENTENCE_SCORE_THRESHOLD = 8.0
MAX_SENTENCE_REFINES = int(os.getenv("MAX_SENTENCE_REFINES", "1"))


def meet_criteria(state: AgentRunningState):
    records = state.get("sentence_records", [])
    if isinstance(records, list):
        eligible_failing_indices = []
        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                continue
            score = rec.get("weighted_score")
            rc = int(rec.get("refine_count", 0) or 0)
            if (
                isinstance(score, (int, float))
                and float(score) < SENTENCE_SCORE_THRESHOLD
                and rc < MAX_SENTENCE_REFINES
            ):
                eligible_failing_indices.append(idx)

        if eligible_failing_indices:
            return "RefinerAgent"
        return "AcceptanceAgent"

    if state["score"] < 8 and state["refine_count"] < MAX_REFINER_ITERATIONS:
        return "RefinerAgent"
    else:
        return "AcceptanceAgent"


def _TaskValidatorPassthrough(state: AgentRunningState):
    return {}


class CodeSwitchingAgent:
    def __init__(self, scenario_k):
        self.state = {}
        self.state["refine_count"] = 0
        self.state["instance_refine_counts"] = []
        for key in scenario_k.keys():
            self.state[key] = scenario_k[key]
        self.state["news_article"] = ""
        self.state["news_hash"] = set()
        self.state["news_dict"] = {}
        self.workflow_with_data_generation: StateGraph = (
            self._construct_graph_with_data_generation()
        )

    def _construct_graph_with_data_generation(self) -> StateGraph:
        workflow = StateGraph(AgentRunningState)
        workflow.add_node("MCPAgent", RunMCPAgent)
        workflow.add_node("DataGenerationAgent", RunDataGenerationAgent)
        workflow.add_node(
            "TaskValidatorAgent",
            RunTaskValidatorAgent if ENABLE_TASK_VALIDATOR else _TaskValidatorPassthrough,
        )
        workflow.add_node("FluencyAgent", RunFluencyAgent)
        workflow.add_node("NaturalnessAgent", RunNaturalnessAgent)
        workflow.add_node("CSRatioAgent", RunCSRatioAgent)
        workflow.add_node("SocialCulturalAgent", RunSocialCulturalAgent)
        workflow.add_node("SummarizeResult", SummarizeResult)
        workflow.add_node("RefinerAgent", RunRefinerAgent)
        workflow.add_node("AcceptanceAgent", AcceptanceAgent)

        # You can change the order of the nodes on demand.
        workflow.add_edge(START, "DataGenerationAgent")
        workflow.add_edge("DataGenerationAgent", "TaskValidatorAgent")
        # MCP is disabled for default.
        # workflow.add_edge("DataGenerationAgent", "MCPAgent")
        # workflow.add_edge("MCPAgent", "FluencyAgent")
        # workflow.add_edge("MCPAgent", "NaturalnessAgent")
        # workflow.add_edge("MCPAgent", "CSRatioAgent")
        # workflow.add_edge("MCPAgent", "SocialCulturalAgent")
        workflow.add_edge("TaskValidatorAgent", "FluencyAgent")
        workflow.add_edge("TaskValidatorAgent", "NaturalnessAgent")
        workflow.add_edge("TaskValidatorAgent", "CSRatioAgent")
        workflow.add_edge("TaskValidatorAgent", "SocialCulturalAgent")
        workflow.add_edge(
            ["FluencyAgent", "NaturalnessAgent", "CSRatioAgent", "SocialCulturalAgent"],
            "SummarizeResult",
        )
        workflow.add_conditional_edges("SummarizeResult", meet_criteria)
        workflow.add_edge("RefinerAgent", "TaskValidatorAgent")
        workflow.add_edge("AcceptanceAgent", END)
        graph = workflow.compile()
        # workflow.add_edge("NewsGenerationAgent", END)
        return graph

    async def run(self):
        # logger.info(f"🤖 Running scenario: {self.scenario_k}")
        try:
            return await self.workflow_with_data_generation.ainvoke(
                self.state, {"recursion_limit": 1e10}
            )
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Scenario timed out after 10 seconds: {self.scenario_k}")
            return ""
        except Exception:
            logger.exception(f"🚨 Scenario failed: {self.state}")
            raise


async def arun(scenario_k):
    agent_instance = CodeSwitchingAgent(scenario_k)
    await agent_instance.run()


async def main():
    config: dict = load_config("../config/config2.yaml")
    scenarios: list[AgentRunningState] = generate_scenarios(config["pre_execute"])
    lang = config["pre_execute"]["shared"]["character_setting"]["nationality"]["first_language"]
    print('scenarios',scenarios)
    print('length senarios', len(scenarios))
    # shuffle scenarios
    random.shuffle(scenarios)
    # make a for loop, each loop run 10 scenarios
    results_count = 0
    failed_count = 0
    for i in range(0, len(scenarios), 1):
        tasks = [arun(scenario) for scenario in scenarios[i : i + 1]]

        # 使用 asyncio.as_completed
        try:
            for task in asyncio.as_completed(tasks, timeout=7200):
                try:
                    await task
                    results_count += 1
                except Exception as e:
                    failed_count += 1
                    logger.exception(f"🚨 Scenario failed at index {i}: {e}")
                    send_message(
                        f"🔍 LANG: {lang} Scenario failed at index {i}: {type(e).__name__}"
                    )
                    continue
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Scenario timed out after 2400 seconds: {i}")
            send_message(
                f"🔍 LANG: {lang} Scenario timed out after 2400 seconds: {i}"
            )
            continue
        finally:
            # log the number of results finished
            if results_count % 200 == 0:
                logger.info(f"🔍 Number of results finished: {results_count}")
                send_message(
                    f"🔍 LANG: {lang} Number of results finished: {results_count}"
                )
    logger.info(
        f"✅ Pipeline completed. Success={results_count}, Failed={failed_count}"
    )
    return results_count


def send_message(message):
    WEBHOOK = None
    params = {"msg_type": "text", "content": {"text": message}}
    import requests
    if WEBHOOK:
        requests.post(WEBHOOK, json=params)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        logger.exception("🚨 Unhandled error while running pipeline")
