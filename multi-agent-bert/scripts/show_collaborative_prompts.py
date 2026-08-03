"""Capture and print the EXACT prompts each agent receives in sequential-
COLLABORATIVE mode (no API cost; recording client). Shows the full prompt for
Lexical, Polarity, Contextual so the collaborative chain block is visible in situ.
"""
from __future__ import annotations
import os, sys, io

sys.path.insert(0, os.path.abspath('.'))
from src.state.schema import PipelineState, StateMetadata, TaskConfig
from src.agents.llm_lexical_agent import LLMLexicalAgent
from src.agents.polarity_agent import PolarityAgent
from src.agents.contextual_agent import ContextualAgent


class RecordingClient:
    """Records every prompt; returns realistic valid JSON so the chain looks real."""
    def __init__(self):
        self.prompts = []
        self._returns = [
            '{"label": "negative", "confidence": 0.80, "reasoning": "The word \'dislike\' is a negative surface cue.", "evidence": ["dislike"]}',
            '{"label": "neutral", "confidence": 0.85, "reasoning": "The author asks a question about others\' dislikes without stating their own view.", "evidence": ["على اساس ايه"]}',
            '{"label": "neutral", "confidence": 0.85, "reasoning": "A rhetorical question, no author evaluation.", "evidence": ["اخوانا"]}',
        ]
        self._n = 0

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        r = self._returns[min(self._n, len(self._returns) - 1)]
        self._n += 1
        return r


tc = TaskConfig(
    task_name="sentiment_classification",
    labels=["positive", "negative", "neutral"],
    label_descriptions={"positive": "praise/approval", "negative": "criticism/dislike",
                        "neutral": "no clear evaluation"},
    sequential_chain=True,
    sequential_chain_style="collaborative",
)
state = PipelineState(
    metadata=StateMetadata(sample_id="demo"),
    input_text="اخوانا اللى عاملين dislike على اساس ايه",
    task_config=tc,
)

client = RecordingClient()
LLMLexicalAgent(llm_client=client).run(state)
PolarityAgent(llm_client=client).run(state)
ContextualAgent(llm_client=client).run(state)

o = io.StringIO()
def P(s=""): o.write(s + "\n")
names = ["LEXICAL (stage 1)", "POLARITY (stage 2)", "CONTEXTUAL (stage 3)"]
for name, prompt in zip(names, client.prompts):
    P("#" * 100)
    P(f"### FULL PROMPT SENT TO: {name}   [sequential_chain_style=collaborative]")
    P("#" * 100)
    P(prompt)
    P("")
open("experiments/outputs/multi_agent_bert/_collaborative_prompts.txt", "w", encoding="utf-8").write(o.getvalue())
print("written")
