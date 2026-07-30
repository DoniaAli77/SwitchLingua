"""scripts/ner_escalation_diff.py

For the sentences that ESCALATE at a given threshold, show per-token:
GOLD, MODEL-ALONE (primary), FUSED (primary+agents), and flag tokens where the
agents changed the model's answer — classifying each such change as Arabic vs
Latin script. Answers: "does escalation hurt only Arabic entities, or English too?"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

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
from src.llm.mock_client import MockLLMClient
from src.models.transformer_ner_tagger import TransformerNERTagger
from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.router import Router
from src.state.schema import ModelOutput, PipelineState, StateMetadata, TaskConfig

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
DATA = Path(__file__).resolve().parent.parent / "data" / "dev_dummy_ner.jsonl"
THRESHOLD = 0.95
MODEL_WEIGHT = 3.0

_GAZ = {"ORG": ["Google", "Microsoft", "Amazon", "Apple", "OpenAI", "Tesla", "UNESCO", "WHO", "FIFA"],
        "LOC": ["Paris", "Cairo", "Dubai", "London", "Berlin", "Geneva", "Riyadh", "Qatar", "Egypt"],
        "PER": ["Ahmed", "Sara", "Mohamed", "Omar", "Fatima", "Ali", "Elon", "Musk"]}
_RULES = {"PER": [r"(Dr|Mr|Ms|Mrs|CEO)\.?"]}
_KNOWN = {"google": "ORG", "microsoft": "ORG", "amazon": "ORG", "cairo": "LOC", "london": "LOC"}


def _is_arabic(tok):
    return any(0x0600 <= ord(c) <= 0x06FF for c in tok)


def _base():
    llm = MockLLMClient(mode="label_echo", allowed_labels=LABELS)
    return dict(primary_classifier=type("P", (), {"run": lambda self, s: (
        setattr(s, "primary_model_output", ModelOutput(label="O", confidence=0.9,
                probabilities={"O": 1.0})) or s)})(),
        router=Router(), lexical_agent=LexicalAgent(),
        contextual_agent=ContextualAgent(llm_client=llm), logic_agent=LogicAgent(),
        consensus_agent=ConsensusAgent(), explainability_agent=ExplainabilityAgent())


def _fused(tagger):
    return PipelineOrchestrator(
        **_base(),
        ner_lexical_agent=NERLexicalAgent(gazetteer=_GAZ),
        ner_logic_agent=NERLogicAgent(rule_map=_RULES),
        ner_contextual_agent=NERContextualAgent(known_entities=_KNOWN),
        ner_consensus_agent=NERConsensusAgent(weights={"model": MODEL_WEIGHT}),
        ner_primary=tagger)


def _state(tokens, mode, threshold):
    return PipelineState(metadata=StateMetadata(sample_id="s"), input_text=" ".join(tokens),
        task_config=TaskConfig(task_name="ner", task_type="sequence_labeling",
            labels=LABELS, pipeline_mode=mode, threshold=threshold), extras={"tokens": tokens})


def main():
    data = [json.loads(l) for l in DATA.open(encoding="utf-8")]
    print("Loading XLM-R NER model ...")
    tagger = TransformerNERTagger.from_pretrained(device="cpu")
    fused = _fused(tagger)

    changed_arabic = changed_latin = 0
    hurt_arabic = hurt_latin = 0
    for d in data:
        tokens, gold = d["tokens"], d["tags"]
        model_only = [t.tag for t in tagger.tag(tokens, task_labels=LABELS).tags]
        if min(t.confidence for t in tagger.tag(tokens, task_labels=LABELS).tags) >= THRESHOLD:
            continue  # would be accepted, not escalated

        res = fused.run(_state(tokens, "paper_style", THRESHOLD))
        fused_tags = [t["tag"] for t in res.final_output.payload["sequence_output"]]

        print("\n" + "=" * 74)
        print(f"{d['id']}: {d['text']}   [decision={res.routing_info.decision}]")
        print("-" * 74)
        print(f"{'WORD':<15}{'script':<8}{'GOLD':<8}{'MODEL':<9}{'FUSED':<9}{'change'}")
        print("-" * 74)
        for w, g, m, f in zip(tokens, gold, model_only, fused_tags):
            script = "AR" if _is_arabic(w) else "lat"
            note = ""
            if m != f:
                note = "CHANGED"
                if _is_arabic(w):
                    changed_arabic += 1
                    if m == g and f != g:
                        hurt_arabic += 1
                else:
                    changed_latin += 1
                    if m == g and f != g:
                        hurt_latin += 1
            print(f"{w:<15}{script:<8}{g:<8}{m:<9}{f:<9}{note}")

    print("\n" + "=" * 74)
    print("SUMMARY of tokens the agents changed vs the model:")
    print(f"  changed  — Arabic: {changed_arabic}   Latin: {changed_latin}")
    print(f"  of which HURT (model was right, agents wrong) — Arabic: {hurt_arabic}   Latin: {hurt_latin}")


if __name__ == "__main__":
    main()
