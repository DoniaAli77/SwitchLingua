"""Proof that the sequential chain is REAL: capture the exact prompt each agent
receives and show that (a) with sequential_chain=True each later agent's prompt
contains the prior agents' conclusions, and (b) with sequential_chain=False no
such block appears. Uses a recording LLM client (no API cost); the prompt-building
code path is identical to the paid runs, so this proves what those runs sent.
"""
from __future__ import annotations
import os, sys, io

sys.path.insert(0, os.path.abspath('.'))
from src.state.schema import PipelineState, StateMetadata, TaskConfig
from src.agents.llm_lexical_agent import LLMLexicalAgent
from src.agents.polarity_agent import PolarityAgent
from src.agents.contextual_agent import ContextualAgent


class RecordingClient:
    """Returns a valid JSON and records every prompt it is asked to generate on."""
    def __init__(self):
        self.prompts = []
        self._n = 0

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        self._n += 1
        # distinctive per-call note so we can trace it flowing into later prompts
        return ('{"label": "negative", "confidence": 0.80, '
                f'"reasoning": "NOTE_FROM_CALL_{self._n}", "evidence": ["cue"]}}')


def make_state(sequential_chain: bool) -> PipelineState:
    return PipelineState(
        metadata=StateMetadata(sample_id="verify-1"),
        input_text="اخوانا اللى عاملين dislike على اساس ايه",
        task_config=TaskConfig(
            task_name="sentiment_classification",
            labels=["positive", "negative", "neutral"],
            label_descriptions={"positive": "pos", "negative": "neg", "neutral": "neu"},
            sequential_chain=sequential_chain,
        ),
    )


def run_chain(sequential_chain: bool):
    client = RecordingClient()
    state = make_state(sequential_chain)
    # Run the three voters in the default chain order, sharing ONE state.
    LLMLexicalAgent(llm_client=client).run(state)
    PolarityAgent(llm_client=client).run(state)      # writes logic_output
    ContextualAgent(llm_client=client).run(state)
    return client.prompts


MARK = "PRIOR AGENTS' ANALYSES"
o = io.StringIO()
def P(s=""): o.write(s + "\n")

for flag in (True, False):
    prompts = run_chain(flag)
    P("=" * 90)
    P(f"sequential_chain = {flag}")
    P("=" * 90)
    for name, pr in zip(["1. LEXICAL", "2. POLARITY", "3. CONTEXTUAL"], prompts):
        has = MARK in pr
        P(f"\n--- {name} agent prompt --- contains prior-agents block? {has}")
        if has:
            # print just the block for readability
            start = pr.index(MARK)
            end = pr.find("\n\n", start)
            P(pr[start:end if end != -1 else start + 400])
        # also confirm whether prior NOTE strings leaked forward
        leaks = [f"NOTE_FROM_CALL_{i}" for i in (1, 2, 3) if f"NOTE_FROM_CALL_{i}" in pr]
        P(f"    prior-call notes present in this prompt: {leaks or 'none'}")
    P("")

# assertions (fail loudly if the chain is not wired as claimed)
seq = run_chain(True)
par = run_chain(False)
assert MARK not in seq[0], "Lexical (first) should have NO prior block"
assert MARK in seq[1], "Polarity should receive the prior block"
assert MARK in seq[2], "Contextual should receive the prior block"
assert "NOTE_FROM_CALL_1" in seq[1], "Polarity prompt must contain Lexical's note"
assert "NOTE_FROM_CALL_1" in seq[2] and "NOTE_FROM_CALL_2" in seq[2], \
    "Contextual prompt must contain Lexical's AND Polarity's notes"
assert all(MARK not in p for p in par), "sequential_chain=False must inject NO block"
P("ALL ASSERTIONS PASSED: chain is real when ON, absent when OFF.")

open("experiments/outputs/multi_agent_bert/_verify_sequential_chain.txt", "w", encoding="utf-8").write(o.getvalue())
print("written; assertions passed")
