"""ner_agentic.py — Canonical builder for the AGENTIC NER pipeline.

One place that defines "the agentic NER setup" so experiments and callers don't
re-wire it each time:

    XLM-R primary  ->  router  ->  (confident? use primary)
                                ->  (unsure? LLM specialist -> consensus)

The rule-based NER agents (gazetteer / regex / capitalisation) are turned OFF
(zero consensus weight), so only the real model and the LLM specialist decide.
The classification-side components the orchestrator constructor requires are
filled with light defaults; they never run on the NER path (task_type routing),
so they are inert here.

Usage
-----
    from src.models.transformer_ner_tagger import TransformerNERTagger
    from src.llm.openai_client import OpenAIClient
    from src.pipeline.ner_agentic import build_agentic_ner_orchestrator, agentic_ner_task_config

    tagger = TransformerNERTagger.from_pretrained(device="cpu")
    orch = build_agentic_ner_orchestrator(tagger, OpenAIClient(), specialist="reflector")
    tc = agentic_ner_task_config(labels=[...], threshold=0.95)
"""

from __future__ import annotations

from typing import List, Optional

from src.agents.consensus_agent import ConsensusAgent
from src.agents.contextual_agent import ContextualAgent
from src.agents.explainability_agent import ExplainabilityAgent
from src.agents.lexical_agent import LexicalAgent
from src.agents.llm_ner_agent import LLMNERAgent
from src.agents.llm_ner_reflection_agent import LLMNERReflectionAgent
from src.agents.logic_agent import LogicAgent
from src.agents.ner_consensus_agent import NERConsensusAgent
from src.llm.base_client import LLMClient
from src.llm.mock_client import MockLLMClient
from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.router import Router
from src.state.schema import ModelOutput, PipelineState, TaskConfig

#: Valid specialist choices for the agentic NER panel.
SPECIALISTS = ("reflector", "tagger")


class _InertPrimary:
    """Placeholder classification primary — never invoked on the NER path.

    The NER path uses ``ner_primary`` (the real model); this only satisfies the
    orchestrator constructor and is a no-op if ever reached.
    """

    def run(self, state: PipelineState) -> PipelineState:  # pragma: no cover - inert
        if state.primary_model_output is None:
            state.primary_model_output = ModelOutput(label=None, confidence=0.0)
        return state


def build_agentic_ner_orchestrator(
    ner_model,
    llm_client: LLMClient,
    specialist: str = "reflector",
    model_weight: float = 1.0,
    llm_weight: float = 1.0,
    logger=None,
) -> PipelineOrchestrator:
    """Build the standard agentic NER orchestrator.

    Parameters
    ----------
    ner_model:
        The NER primary (e.g. ``TransformerNERTagger``) — runs first, its output
        drives the router.
    llm_client:
        LLM backend for the specialist (e.g. ``OpenAIClient``; ``MockLLMClient``
        in tests).
    specialist:
        ``"reflector"`` (LLM reviews & corrects the primary's draft — default) or
        ``"tagger"`` (LLM tags independently and votes).
    model_weight, llm_weight:
        Consensus weights for the primary (``model`` slot) and the LLM
        (``contextual`` slot). The rule agents (``lexical``/``logic``) are fixed
        at 0 so they never influence the result.
    """
    if specialist not in SPECIALISTS:
        raise ValueError(f"specialist must be one of {SPECIALISTS}, got '{specialist}'.")

    if specialist == "reflector":
        ner_specialist = LLMNERReflectionAgent(llm_client, output_slot="contextual")
    else:
        ner_specialist = LLMNERAgent(llm_client, output_slot="contextual")

    # Consensus: only the model (primary) and contextual (LLM) vote; rules OFF.
    consensus = NERConsensusAgent(weights={
        "model": model_weight, "contextual": llm_weight,
        "lexical": 0.0, "logic": 0.0,
    })

    # Inert classification-side components (never run on the NER path).
    cls_llm = MockLLMClient(mode="label_echo", allowed_labels=["_"])
    return PipelineOrchestrator(
        primary_classifier=_InertPrimary(),
        router=Router(),
        lexical_agent=LexicalAgent(),
        contextual_agent=ContextualAgent(llm_client=cls_llm),
        logic_agent=LogicAgent(),
        consensus_agent=ConsensusAgent(),
        explainability_agent=ExplainabilityAgent(),
        ner_contextual_agent=ner_specialist,   # LLM specialist occupies this slot
        ner_consensus_agent=consensus,
        ner_primary=ner_model,
        logger=logger,
    )


def agentic_ner_task_config(
    labels: List[str],
    threshold: float = 0.95,
    pipeline_mode: str = "full_agentic",
    task_name: str = "ner",
    label_descriptions: Optional[dict] = None,
) -> TaskConfig:
    """Return a TaskConfig for the agentic NER path (sequence_labeling)."""
    return TaskConfig(
        task_name=task_name,
        task_type="sequence_labeling",
        labels=labels,
        label_descriptions=label_descriptions or {},
        threshold=threshold,
        pipeline_mode=pipeline_mode,  # type: ignore[arg-type]
    )
