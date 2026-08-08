"""llm_ner_panel_agents.py — the two agent ROLES from the multi-agent NER
literature that our pipeline was missing:

* :class:`LLMNERDebateAgent`   — CROSSAGENTIE-style. Given two taggers' outputs
  (e.g. the primary model and a specialist), it resolves ONLY the tokens where
  they disagree, via an LLM "judge". Agreement positions are left untouched.

* :class:`LLMNERDisambiguationAgent` — KDR-Agent-style. Given a draft, it
  re-decides the TYPE of already-detected entities (PERS vs LOC vs ORG vs MISC)
  using sentence context. It does NOT add or delete entities — a narrow
  type-resolver, distinct from the reflector (which does everything).

Both reuse the JSON parsing / length-alignment / tag-coercion helpers from
:mod:`src.agents.llm_ner_agent`. Sequence-labeling only; classification untouched.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from src.agents.llm_ner_agent import (
    _LLM_CONFIDENCE,
    _SLOT_ATTR,
    LLMNERAgent,
    LLMNERParseError,
    coerce_to_valid,
)
from src.state.schema import (
    AgentOutput,
    ModelOutput,
    PipelineState,
    SequenceLabelingOutput,
    TokenTag,
)

# Readable sources include the specialist slots plus the primary model's slot.
_SRC_ATTR = {**_SLOT_ATTR, "model": "ner_model_output"}


def _read_tags(state: PipelineState, slot: str) -> Optional[List[str]]:
    """Return the per-token tag strings held in *slot*, or None if absent."""
    ao = getattr(state, _SRC_ATTR.get(slot, ""), None)
    if ao is None or ao.sequence_output is None:
        return None
    return [tt.tag for tt in ao.sequence_output.tags]


def _pad(tags: List[str], n: int) -> List[str]:
    return (list(tags) + ["O"] * n)[:n]


class _BasePanelAgent(LLMNERAgent):
    """Shared write/skip plumbing for the panel agents."""

    def _write(self, state, tokens, tags, note, extra=None):
        seq = SequenceLabelingOutput(
            tags=[TokenTag(token=t, tag=g, confidence=_LLM_CONFIDENCE)
                  for t, g in zip(tokens, tags)],
            notes=note, features=extra or {})
        setattr(state, _SLOT_ATTR[self.output_slot], AgentOutput(
            agent_name=self.name, model_output=ModelOutput(),
            sequence_output=seq, notes=note))
        state.append_history(
            component=self.name, summary=f"{self.name}: {note}"[:200],
            outputs={"tags": [f"{t}:{g}" for t, g in zip(tokens, tags)][:60]})

    def _skip_or_none(self, state) -> bool:
        if state.task_config.task_type != "sequence_labeling":
            state.append_history(component=self.name,
                                 summary=f"{self.name}: not sequence_labeling — skipped.")
            return True
        return False


# ---------------------------------------------------------------------------
# Debate agent
# ---------------------------------------------------------------------------

class LLMNERDebateAgent(_BasePanelAgent):
    """Resolve tokens where two taggers disagree (CROSSAGENTIE-style judge)."""

    def __init__(self, llm_client, source_a: str = "model", source_b: str = "contextual",
                 output_slot: str = "contextual", name=None, logger=None):
        super().__init__(llm_client, output_slot=output_slot,
                         name=name or "LLMNERDebateAgent", logger=logger)
        self.source_a, self.source_b = source_a, source_b

    def run(self, state: PipelineState) -> PipelineState:
        if self._skip_or_none(state):
            return state
        task = state.task_config
        tokens: List[str] = state.extras.get("tokens") or state.input_text.split()
        n = len(tokens)
        a, b = _read_tags(state, self.source_a), _read_tags(state, self.source_b)
        if a is None or b is None:            # nothing to debate; pass through
            src = a if a is not None else b
            if src is not None:
                self._write(state, tokens, _pad(src, n), "no counterpart to debate")
            return state
        ta, tb = _pad(a, n), _pad(b, n)
        disagree = [i for i in range(n) if ta[i] != tb[i]]
        if not disagree:
            self._write(state, tokens, ta, "no disagreements", {"disagreements": 0})
            return state

        raw = self.llm_client.generate(self._prompt(task, tokens, ta, tb, disagree))
        try:
            llm_tags, reasoning = self._parse_response(raw, n)
            note = ""
        except LLMNERParseError as exc:
            self.logger.warning("%s: parse error — %s; kept source A.", self.name, exc)
            llm_tags, reasoning, note = ta, "", f"debate parse fail ({exc}); kept source A"
        llm_tags, _ = self._align_length(llm_tags, n)
        valid = set(task.labels)
        final = list(ta)
        for i in disagree:                    # only touch the contested positions
            final[i] = coerce_to_valid(llm_tags[i], valid)
        self._write(state, tokens, final, note or reasoning or "debated",
                    {"disagreements": len(disagree)})
        return state

    @staticmethod
    def _prompt(task, tokens, ta, tb, disagree) -> str:
        lines = []
        for i, tok in enumerate(tokens):
            if i in disagree:
                lines.append(f"  {i}: {tok}   [A={ta[i]} vs B={tb[i]}]  <-- DECIDE")
            else:
                lines.append(f"  {i}: {tok}   [{ta[i]}]")
        allowed = ", ".join(task.labels)
        return (
            "You are the JUDGE in a multi-agent NER system for code-switched "
            "Arabic-English text. Two taggers (A and B) disagree on some tokens.\n"
            f"Allowed tags: {allowed}\n\n"
            f"SENTENCE: {' '.join(tokens)}\n\n"
            "TOKENS (index: token [current] or [A vs B] for contested):\n"
            + "\n".join(lines) +
            "\n\nFor each CONTESTED token pick the correct tag (A, B, or a better "
            "allowed tag); keep the others as shown. Arabic-script names are "
            "entities too. Respond with JSON only: "
            '{\"tags\": [<one allowed tag per token, in order>], \"reasoning\": \"<one sentence>\"}'
        )


# ---------------------------------------------------------------------------
# Disambiguation agent
# ---------------------------------------------------------------------------

class LLMNERDisambiguationAgent(_BasePanelAgent):
    """Re-decide the TYPE of already-detected entities (KDR-Agent-style)."""

    def __init__(self, llm_client, source_slot: str = "contextual",
                 output_slot: str = "contextual", name=None, logger=None):
        super().__init__(llm_client, output_slot=output_slot,
                         name=name or "LLMNERDisambiguationAgent", logger=logger)
        self.source_slot = source_slot

    def run(self, state: PipelineState) -> PipelineState:
        if self._skip_or_none(state):
            return state
        task = state.task_config
        tokens: List[str] = state.extras.get("tokens") or state.input_text.split()
        n = len(tokens)
        cur = _read_tags(state, self.source_slot)
        if cur is None:
            cur = _read_tags(state, "model")
        if cur is None:
            return state
        tags = _pad(cur, n)
        ent_pos = [i for i in range(n) if tags[i] != "O"]
        if not ent_pos:
            self._write(state, tokens, tags, "no entities to disambiguate",
                        {"entities": 0})
            return state

        raw = self.llm_client.generate(self._prompt(task, tokens, tags, ent_pos))
        try:
            llm_tags, reasoning = self._parse_response(raw, n)
            note = ""
        except LLMNERParseError as exc:
            self.logger.warning("%s: parse error — %s; kept draft.", self.name, exc)
            llm_tags, reasoning, note = tags, "", f"disambig parse fail ({exc}); kept draft"
        llm_tags, _ = self._align_length(llm_tags, n)
        valid = set(task.labels)
        final = list(tags)
        changed = 0
        for i in ent_pos:                     # only re-type entities, never delete/add
            new = coerce_to_valid(llm_tags[i], valid)
            if new != "O" and new != final[i]:
                final[i] = new
                changed += 1
        self._write(state, tokens, final, note or reasoning or "disambiguated",
                    {"entities": len(ent_pos), "retyped": changed})
        return state

    @staticmethod
    def _prompt(task, tokens, tags, ent_pos) -> str:
        ent_types = [l for l in task.labels if l != "O"]
        lines = []
        for i, tok in enumerate(tokens):
            mark = "  <-- CONFIRM/CORRECT TYPE" if i in ent_pos else ""
            lines.append(f"  {i}: {tok}   [{tags[i]}]{mark}")
        allowed = ", ".join(task.labels)
        types = ", ".join(ent_types)
        return (
            "You are the DISAMBIGUATION agent in a multi-agent NER system for "
            "code-switched Arabic-English text. Some tokens are already marked as "
            "entities. Your ONLY job: for each marked token, confirm or correct its "
            f"TYPE (one of: {types}) using sentence context. Do NOT change tokens "
            "that are not marked, and do NOT remove entities.\n"
            f"Allowed tags: {allowed}\n\n"
            f"SENTENCE: {' '.join(tokens)}\n\n"
            "TOKENS:\n" + "\n".join(lines) +
            "\n\nRespond with JSON only: "
            '{\"tags\": [<one allowed tag per token, in order>], \"reasoning\": \"<one sentence>\"}'
        )
