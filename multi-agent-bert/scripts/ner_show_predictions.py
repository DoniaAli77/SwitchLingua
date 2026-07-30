"""scripts/ner_show_predictions.py

Print per-word NER predictions for a few Arabic-English sentences, comparing
GOLD vs the heuristic rules vs the XLM-R model, so the difference is visible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Windows consoles default to cp1252 which cannot print Arabic — force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.transformer_ner_tagger import TransformerNERTagger
from src.state.schema import (
    AgentOutput, ModelOutput, PipelineState, SequenceLabelingOutput,
    StateMetadata, TaskConfig, TokenTag,
)
from src.agents.ner_lexical_agent import NERLexicalAgent
from src.agents.ner_logic_agent import NERLogicAgent
from src.agents.ner_contextual_agent import NERContextualAgent
from src.agents.ner_consensus_agent import NERConsensusAgent

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
DATA = Path(__file__).resolve().parent.parent / "data" / "dev_dummy_ner.jsonl"

# Hand-authored heuristic resources (general knowledge, not gold).
_GAZ = {"ORG": ["Google", "Microsoft", "Amazon", "Apple", "OpenAI", "Tesla", "UNESCO", "WHO", "FIFA"],
        "LOC": ["Paris", "Cairo", "Dubai", "London", "Berlin", "Geneva", "Riyadh", "Qatar", "Egypt"],
        "PER": ["Ahmed", "Sara", "Mohamed", "Omar", "Fatima", "Ali", "Elon", "Musk"]}
_KNOWN = {"google": "ORG", "microsoft": "ORG", "amazon": "ORG", "cairo": "LOC", "london": "LOC"}

# A few sentences with Arabic + English entities.
SHOW_IDS = ["ner_02", "ner_28", "ner_13"]


def _state(tokens):
    return PipelineState(
        metadata=StateMetadata(sample_id="s"), input_text=" ".join(tokens),
        task_config=TaskConfig(task_name="ner", task_type="sequence_labeling", labels=LABELS),
        extras={"tokens": tokens})


def _rules_tags(tokens):
    """Run the three heuristic agents + consensus, return per-token tags."""
    st = _state(tokens)
    NERLexicalAgent(gazetteer=_GAZ).run(st)
    NERLogicAgent().run(st)
    NERContextualAgent(known_entities=_KNOWN).run(st)
    NERConsensusAgent().run(st)
    return [t["tag"] for t in st.final_output.payload["sequence_output"]]


def main():
    data = {d["id"]: d for d in (json.loads(l) for l in DATA.open(encoding="utf-8"))}
    print("Loading XLM-R NER model ...")
    tagger = TransformerNERTagger.from_pretrained(device="cpu")

    for sid in SHOW_IDS:
        d = data[sid]
        tokens, gold = d["tokens"], d["tags"]
        rules = _rules_tags(tokens)
        xlmr = [t.tag for t in tagger.tag(tokens, task_labels=LABELS).tags]

        print("\n" + "=" * 66)
        print(f"{sid}:  {d['text']}")
        print("-" * 66)
        print(f"{'WORD':<16}{'GOLD':<9}{'RULES':<12}{'XLM-R':<12}")
        print("-" * 66)
        for w, g, r, x in zip(tokens, gold, rules, xlmr):
            rmark = "OK " if r == g else "xx "
            xmark = "OK " if x == g else "xx "
            print(f"{w:<16}{g:<9}{rmark+r:<12}{xmark+x:<12}")
        r_ok = sum(r == g for r, g in zip(rules, gold))
        x_ok = sum(x == g for x, g in zip(xlmr, gold))
        n = len(gold)
        print("-" * 66)
        print(f"{'correct:':<16}{'':<9}{f'{r_ok}/{n}':<12}{f'{x_ok}/{n}':<12}")


if __name__ == "__main__":
    main()
