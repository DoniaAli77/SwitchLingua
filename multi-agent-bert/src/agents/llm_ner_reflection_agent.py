"""LLM-based NER REFLECTION agent (review-and-correct).

Unlike :class:`~src.agents.llm_ner_agent.LLMNERAgent` (which tags from scratch
and then votes), this agent reads the PRIMARY model's draft tags from
``state.ner_model_output`` and asks the LLM to CORRECT ONLY THE MISTAKES. Because
it starts from the strong primary and only fixes errors, it can help without
overriding the primary's already-correct predictions — the mechanism that makes
agents improve NER in recent multi-agent work (KDR-Agent, CROSSAGENTIE).

It reuses the base agent's JSON parsing / length-alignment helpers and writes a
corrected :class:`~src.state.schema.SequenceLabelingOutput` to the chosen slot
(default ``contextual_output``). Run it as the sole specialist with consensus
weighted only on its slot, so its corrected output becomes the final answer.
"""

from __future__ import annotations

from typing import List, Optional

from src.agents.llm_ner_agent import (
    _LLM_CONFIDENCE,
    _SLOT_ATTR,
    LLMNERAgent,
    LLMNERParseError,
    coerce_to_valid,
)
from src.prompts.llm_ner_reflection_prompt import (
    build_reflection_user_prompt,
    get_reflection_system_prompt,
)
from src.state.schema import (
    AgentOutput,
    ModelOutput,
    PipelineState,
    SequenceLabelingOutput,
    TokenTag,
)

_SKIP_NOTE = "LLMNERReflectionAgent: task_type is not sequence_labeling — skipped."
_PARSE_FAIL_NOTE = "LLMNERReflectionAgent: parse failed; kept primary draft."


class LLMNERReflectionAgent(LLMNERAgent):
    """LLM reviewer that corrects the primary model's draft NER tags."""

    def __init__(self, llm_client, output_slot: str = "contextual", name=None, logger=None):
        super().__init__(llm_client=llm_client, output_slot=output_slot,
                         name=name or "LLMNERReflectionAgent", logger=logger)

    def _draft_tags(self, state: PipelineState, n_tokens: int) -> List[str]:
        """Read the primary model's per-token draft tags, or all-O if absent."""
        out = state.ner_model_output
        if out is not None and out.sequence_output and out.sequence_output.tags:
            tags = [tt.tag for tt in out.sequence_output.tags]
            return (tags + ["O"] * n_tokens)[:n_tokens]
        return ["O"] * n_tokens

    def run(self, state: PipelineState) -> PipelineState:
        """Correct the primary draft via the LLM; write to the chosen slot."""
        if state.task_config.task_type != "sequence_labeling":
            self.logger.debug(_SKIP_NOTE)
            state.append_history(component=self.name, summary=_SKIP_NOTE)
            return state

        task = state.task_config
        tokens: List[str] = state.extras.get("tokens") or state.input_text.split()
        draft = self._draft_tags(state, len(tokens))

        user_prompt = build_reflection_user_prompt(
            task_name=task.task_name, labels=task.labels,
            tokens=tokens, text=state.input_text, draft_tags=draft,
        )
        system_prompt = get_reflection_system_prompt(
            labels=task.labels, label_descriptions=task.label_descriptions,
        )
        self.logger.debug("%s: reviewing %d-token draft", self.name, len(tokens))
        raw = self.llm_client.generate(f"{system_prompt}\n\n{user_prompt}")

        try:
            tags, reasoning = self._parse_response(raw, len(tokens))
            note = ""
        except LLMNERParseError as exc:
            # On failure, keep the primary's draft unchanged (safe fallback).
            self.logger.warning("%s: parse error — %s; keeping draft.", self.name, exc)
            tags, reasoning, note = list(draft), "", f"{_PARSE_FAIL_NOTE} ({exc})"

        tags, len_note = self._align_length(tags, len(tokens))
        valid = set(task.labels)
        corrected = [coerce_to_valid(t, valid) for t in tags]
        note = " ".join(n for n in (note, len_note) if n).strip()

        n_changed = sum(1 for d, c in zip(draft, corrected) if d != c)
        seq_out = SequenceLabelingOutput(
            tags=[TokenTag(token=tok, tag=tag, confidence=_LLM_CONFIDENCE)
                  for tok, tag in zip(tokens, corrected)],
            notes=note or reasoning,
            features={"raw_llm_response": raw, "reasoning": reasoning,
                      "draft_tags": draft, "tokens_changed": n_changed},
        )
        setattr(state, _SLOT_ATTR[self.output_slot], AgentOutput(
            agent_name=self.name, model_output=ModelOutput(),
            sequence_output=seq_out, notes=seq_out.notes))

        state.append_history(
            component=self.name,
            summary=f"Reflection complete ({self.output_slot} slot): "
                    f"{n_changed}/{len(tokens)} token(s) changed. {reasoning}".strip(),
            outputs={"tokens_changed": n_changed, "reasoning": reasoning,
                     "fallback": bool(note)},
        )
        return state
