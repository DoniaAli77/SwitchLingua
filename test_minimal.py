import asyncio
import sys
import os

# Set OpenAI API credentials

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from utils import load_config, generate_scenarios
from node_engine import (
    RunDataGenerationAgent,
    RunFluencyAgent,
    RunNaturalnessAgent,
    RunCSRatioAgent,
    RunSocialCulturalAgent,
    SummarizeResult,
)
from node_models import AgentRunningState
from loguru import logger
from datetime import datetime

logger.add(f"logs/test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

async def test_minimal_run():
    """Test a single scenario with real config"""
    
    # Load config
    config = load_config("config/config.yaml")
    logger.info(f"✅ Loaded config from config/config.yaml")
    
    # Generate scenarios
    scenarios = generate_scenarios(config["pre_execute"])
    logger.info(f"✅ Generated {len(scenarios)} scenario(s)")
    
    if not scenarios:
        logger.error("❌ No scenarios generated")
        return
    
    # Run first scenario
    scenario = scenarios[0]
    logger.info(f"🚀 Running scenario: {scenario}")
    
    # Initialize state
    state = AgentRunningState()
    state["refine_count"] = 0
    for key, value in scenario.items():
        state[key] = value
    state["news_article"] = ""
    state["news_hash"] = set()
    state["news_dict"] = {}
    state["mcp_result"] = {}
    
    try:
        # Run Data Generation
        logger.info("📝 Running DataGenerationAgent...")
        result = RunDataGenerationAgent(state)
        state.update(result)
        logger.info(f"✅ Generated: {state['data_generation_result']}")
        
        # Run Fluency
        logger.info("🔍 Running FluencyAgent...")
        result = RunFluencyAgent(state)
        state.update(result)
        logger.info(f"✅ Fluency: {state['fluency_result']}")
        
        # Run Naturalness
        logger.info("🔍 Running NaturalnessAgent...")
        result = RunNaturalnessAgent(state)
        state.update(result)
        logger.info(f"✅ Naturalness: {state['naturalness_result']}")
        
        # Run CS Ratio
        logger.info("🔍 Running CSRatioAgent...")
        result = RunCSRatioAgent(state)
        state.update(result)
        logger.info(f"✅ CS Ratio: {state['cs_ratio_result']}")
        
        # Run Socio-Cultural
        logger.info("🔍 Running SocialCulturalAgent...")
        result = RunSocialCulturalAgent(state)
        state.update(result)
        logger.info(f"✅ Socio-Cultural: {state['social_cultural_result']}")
        
        # Summarize
        logger.info("📊 Running SummarizeResult...")
        result = SummarizeResult(state)
        state.update(result)
        logger.info(f"✅ Final Score: {state['score']}")
        
        logger.info("=" * 60)
        logger.info("🎉 TEST PASSED - All agents ran successfully!")
        logger.info(f"Final Score: {state['score']:.2f}/10")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Error during execution: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(test_minimal_run())
