"""Demo script: run the full multi-agent classification pipeline on sample inputs.

Usage (from the multi-agent-bert/ directory):
    python run_demo.py

Two scenarios are exercised:
  1. High-confidence primary prediction → accepted, no escalation.
  2. Low-confidence primary prediction  → escalated through all specialist agents.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Make sure src/ is on the path when running from project root.
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.consensus_agent import ConsensusAgent
from src.agents.contextual_agent import ContextualAgent
from src.agents.explainability_agent import ExplainabilityAgent
from src.agents.lexical_agent import LexicalAgent
from src.agents.logic_agent import LogicAgent
from src.llm.mock_client import MockLLMClient
from src.models.mock_primary_classifier import MockPrimaryClassifier
from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.router import Router
from src.state.schema import PipelineState, StateMetadata, TaskConfig, utc_now

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_demo")

# ---------------------------------------------------------------------------
# Task & label configuration
# ---------------------------------------------------------------------------
LABELS = ["positive", "negative", "neutral"]
LABEL_DESCRIPTIONS = {
    "positive": "Text that expresses a positive or favorable sentiment.",
    "negative": "Text that expresses a negative or unfavorable sentiment.",
    "neutral":  "Text that is factual or carries no strong sentiment.",
}
TASK_NAME = "sentiment_classification"
THRESHOLD = 0.65   # primary confidence must exceed this to skip escalation

# ---------------------------------------------------------------------------
# Keyword / rule configuration (minimal demo values)
# ---------------------------------------------------------------------------
KEYWORD_MAP = {
    "positive": ["great", "excellent", "love", "amazing", "good"],
    "negative": ["terrible", "awful", "hate", "bad", "poor"],
    "neutral":  ["okay", "average", "fine", "normal"],
}

RULE_MAP = {
    "positive": [r"\b(great|excellent|amazing|love)\b"],
    "negative": [r"\b(terrible|awful|hate|bad)\b"],
    "neutral":  [r"\b(okay|average|fine)\b"],
}


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_orchestrator(primary_label: str, primary_conf: float) -> PipelineOrchestrator:
    """Construct a fully wired orchestrator for a single demo run."""
    llm_client = MockLLMClient(
        mode="fixed",
        fixed_response=json.dumps({
            "label": primary_label,
            "confidence": primary_conf,
            "reasoning": "Mock LLM: label chosen for demo purposes.",
            "evidence": ["demo"],
        }),
    )
    return PipelineOrchestrator(
        primary_classifier=MockPrimaryClassifier(
            mode="fixed",
            fixed_label=primary_label,
            fixed_confidence=primary_conf,
        ),
        router=Router(),
        lexical_agent=LexicalAgent(keyword_map=KEYWORD_MAP),
        contextual_agent=ContextualAgent(llm_client=llm_client),
        logic_agent=LogicAgent(rule_map=RULE_MAP),
        consensus_agent=ConsensusAgent(),
        explainability_agent=ExplainabilityAgent(),
    )


def build_state(sample_id: str, text: str) -> PipelineState:
    return PipelineState(
        metadata=StateMetadata(sample_id=sample_id, timestamp=utc_now()),
        input_text=text,
        task_config=TaskConfig(
            task_name=TASK_NAME,
            labels=list(LABELS),
            label_descriptions=dict(LABEL_DESCRIPTIONS),
            threshold=THRESHOLD,
        ),
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _sep(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def _print_result(state: PipelineState) -> None:
    if state.extras.get("pipeline_error"):
        err = state.extras["pipeline_error"]
        print(f"  [ERROR in stage '{err['stage']}'] {err['message']}")
        return

    fo = state.final_output
    co = state.consensus_output
    eo = state.explanation_output
    ri = state.routing_info

    print(f"  Routing decision : {ri.decision if ri else 'n/a'}")
    print(f"  Final label      : {fo.label if fo else 'n/a'}")
    print(f"  Final confidence : {fo.confidence:.3f}" if fo and fo.confidence is not None else "  Final confidence : n/a")

    if co:
        print(f"  Consensus votes  : {co.votes}")

    if eo:
        print(f"\n  Summary   : {eo.summary}")
        if eo.evidence:
            print("  Evidence  :")
            for e in eo.evidence:
                print(f"    • {e}")
        if eo.caveats:
            print("  Caveats   :")
            for c in eo.caveats:
                print(f"    ⚠ {c}")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def scenario_accepted() -> None:
    """Primary model is very confident → fast path, no specialist agents."""
    _sep("Scenario 1 — High confidence (accepted, no escalation)")
    text = "This product is absolutely amazing — I love it!"
    log.info("Input text: %r", text)
    state = build_state("demo-001", text)
    state = build_orchestrator(primary_label="positive", primary_conf=0.92).run(state)
    _print_result(state)


def scenario_escalated_agreement() -> None:
    """Primary model is uncertain → escalated; all agents agree."""
    _sep("Scenario 2 — Low confidence (escalated, full agreement)")
    text = "The product seems okay but nothing great really."
    log.info("Input text: %r", text)
    state = build_state("demo-002", text)
    state = build_orchestrator(primary_label="neutral", primary_conf=0.51).run(state)
    _print_result(state)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scenario_accepted()
    scenario_escalated_agreement()
    print()
