"""scripts/try_synthetic_ner.py

Run the multi-agent NER pipeline on the synthetic Arabic-English code-switched
NER data (data/dev_dummy_ner.jsonl) and print token accuracy + macro-F1.

Two honest configurations are evaluated (no peeking at gold labels):

  A. DEFAULTS  — empty gazetteer, empty regex rules; only the contextual
                 agent's built-in capitalisation heuristic fires.
  B. CONFIGURED — a small, hand-authored, GENERAL-KNOWLEDGE gazetteer /
                 rules / known-entities map (famous orgs, cities, honorifics).
                 These are written from world knowledge, NOT copied from the
                 dataset's gold tags.

Usage
-----
    python scripts/try_synthetic_ner.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.consensus_agent import ConsensusAgent
from src.agents.contextual_agent import ContextualAgent
from src.agents.explainability_agent import ExplainabilityAgent
from src.agents.lexical_agent import LexicalAgent
from src.agents.logic_agent import LogicAgent
from src.agents.ner_consensus_agent import NERConsensusAgent
from src.agents.ner_contextual_agent import NERContextualAgent
from src.agents.ner_lexical_agent import NERLexicalAgent
from src.agents.ner_logic_agent import NERLogicAgent
from src.evaluation.ner_evaluator import NEREvaluator
from src.llm.mock_client import MockLLMClient
from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.router import Router
from src.state.schema import ModelOutput, TaskConfig

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
DATA = Path(__file__).resolve().parent.parent / "data" / "dev_dummy_ner.jsonl"


class _StubPrimary:
    """NER path ignores the primary label; this just avoids a crash."""
    def run(self, state):
        state.primary_model_output = ModelOutput(
            label="O", confidence=0.9, probabilities={"O": 1.0}
        )
        return state


# --- Config B resources: hand-authored from GENERAL knowledge (not gold) ----
_GAZETTEER = {
    "ORG": ["Google", "Microsoft", "Amazon", "Apple", "OpenAI", "Tesla",
            "UNESCO", "WHO", "FIFA", "NASA"],
    "LOC": ["Paris", "Cairo", "Dubai", "London", "Berlin", "Geneva",
            "Riyadh", "Qatar", "Egypt", "Sudan", "Casablanca", "Mars"],
    "PER": ["Ahmed", "Sara", "Mohamed", "Omar", "Fatima", "Ali", "Elon", "Musk"],
}
_RULES = {
    # Honorific followed by a name is a strong PER cue.
    "PER": [r"(Dr|Mr|Ms|Mrs|CEO)\.?"],
}
_KNOWN = {
    "google": "ORG", "microsoft": "ORG", "amazon": "ORG", "apple": "ORG",
    "openai": "ORG", "tesla": "ORG", "unesco": "ORG", "who": "ORG",
    "paris": "LOC", "cairo": "LOC", "dubai": "LOC", "london": "LOC",
    "berlin": "LOC", "geneva": "LOC", "qatar": "LOC",
}


def _make_orch(configured: bool) -> PipelineOrchestrator:
    llm = MockLLMClient(mode="label_echo", allowed_labels=LABELS)
    if configured:
        ner_lex = NERLexicalAgent(gazetteer=_GAZETTEER)
        ner_log = NERLogicAgent(rule_map=_RULES)
        ner_ctx = NERContextualAgent(known_entities=_KNOWN)
    else:
        ner_lex = NERLexicalAgent()          # empty gazetteer -> all O
        ner_log = NERLogicAgent()            # empty rules -> all O
        ner_ctx = NERContextualAgent()       # capitalisation heuristic only
    return PipelineOrchestrator(
        primary_classifier=_StubPrimary(),
        router=Router(),
        lexical_agent=LexicalAgent(),
        contextual_agent=ContextualAgent(llm_client=llm),
        logic_agent=LogicAgent(),
        consensus_agent=ConsensusAgent(),
        explainability_agent=ExplainabilityAgent(),
        ner_lexical_agent=ner_lex,
        ner_logic_agent=ner_log,
        ner_contextual_agent=ner_ctx,
        ner_consensus_agent=NERConsensusAgent(),
    )


def _run(name: str, configured: bool, data):
    tc = TaskConfig(
        task_name="ner",
        task_type="sequence_labeling",
        labels=LABELS,
        pipeline_mode="paper_style",
    )
    ev = NEREvaluator(task_config=tc, orchestrator=_make_orch(configured), run_id=name)
    rep = ev.evaluate(data)
    print(f"\n=== {name} ===")
    print(f"  samples={rep.num_samples}  tokens={rep.num_tokens}  errors={rep.meta['error_samples']}")
    print(f"  token_accuracy = {rep.token_accuracy:.3f}")
    print(f"  macro_f1       = {rep.macro_f1:.3f}")
    print("  per-tag F1:")
    for m in rep.per_tag:
        print(f"    {m.label:6s}  P={m.precision:.2f} R={m.recall:.2f} F1={m.f1:.2f}  (support={m.support})")
    return rep


def main():
    data = [json.loads(l) for l in DATA.open(encoding="utf-8")]
    _run("A_DEFAULTS", configured=False, data=data)
    _run("B_CONFIGURED", configured=True, data=data)


if __name__ == "__main__":
    main()
