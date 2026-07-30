"""LLM-based NER specialist agent for the full_agentic pipeline mode.

The sequence-labeling counterpart of :class:`~src.agents.llm_lexical_agent.LLMLexicalAgent`.
Where the heuristic NER agents (gazetteer / regex / capitalisation) are blind to
Arabic-script entities, this agent delegates per-token BIO tagging to an LLM
(e.g. gpt-4o-mini via :class:`~src.llm.openai_client.OpenAIClient`), which reads
both Arabic and English. It is a drop-in replacement for a heuristic NER
specialist: it writes a :class:`~src.state.schema.SequenceLabelingOutput` to a
chosen state slot (default ``contextual_output``), so the existing
:class:`~src.agents.ner_consensus_agent.NERConsensusAgent` votes on it unchanged.

Robustness
----------
The LLM can return the wrong number of tags or an invalid tag. The agent:
* pads / truncates the tag list to the token count (recording a note), and
* maps any out-of-vocabulary tag to ``"O"``.
On an unparseable response it falls back to an all-``"O"`` output (recording a
note) rather than crashing the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.llm.base_client import LLMClient, LLMClientError  # noqa: F401
from src.prompts.llm_ner_prompt import build_user_prompt, get_system_prompt
from src.state.schema import (
    AgentOutput,
    ModelOutput,
    PipelineState,
    SequenceLabelingOutput,
    TokenTag,
)

_SKIP_NOTE = "LLMNERAgent: task_type is not sequence_labeling — skipped."
_PARSE_FAIL_NOTE = "LLMNERAgent: response parse failed; all-O fallback applied."
_LEN_MISMATCH_NOTE = "LLMNERAgent: tag count != token count; padded/truncated."

# State attribute each logical slot writes to (matches NERConsensusAgent slots).
_SLOT_ATTR = {
    "contextual": "contextual_output",
    "lexical": "lexical_output",
    "logic": "logic_output",
}

# Confidence assigned to LLM per-token tags (the LLM does not emit per-token
# probabilities; a single high value lets the consensus weight it sensibly).
_LLM_CONFIDENCE = 0.9


# Entity-type synonyms — LLMs emit standard CoNLL types (PER/ORG/LOC/MISC) which
# may differ from a corpus's label names (e.g. Sabty uses PERS). Grouped so a
# returned type can be mapped to whichever synonym the task actually uses.
_ALIAS_GROUPS = [
    {"PER", "PERS", "PERSON", "PEOPLE"},
    {"ORG", "ORGANISATION", "ORGANIZATION"},
    {"LOC", "LOCATION", "GPE", "PLACE"},
    {"MISC", "MISCELLANEOUS", "OTHER"},
]


def _resolve_alias(etype: str, valid: set):
    """Return the member of *etype*'s synonym group that is in *valid*, else None."""
    up = etype.upper()
    if up in valid:
        return up
    for grp in _ALIAS_GROUPS:
        if up in grp:
            for member in grp:
                if member in valid:
                    return member
    return None


def coerce_to_valid(tag: str, valid: set) -> str:
    """Map an LLM-returned tag onto the task's label set.

    Handles two mismatches: (1) BIO vs type-level scheme — the prompt may elicit
    ``B-PER`` while the task uses bare types; (2) type-name synonyms — the LLM
    emits standard ``PER`` while the corpus uses ``PERS``. When the tag itself is
    not valid but its (aliased) entity type is, that type is used. Anything
    unrecognised -> ``"O"``.
    """
    if tag in valid:
        return tag
    etype = tag.split("-", 1)[1] if "-" in tag else tag
    if etype in valid:
        return etype
    resolved = _resolve_alias(etype, valid)
    return resolved if resolved is not None else "O"


class LLMNERParseError(ValueError):
    """Raised when the LLM NER response cannot be parsed into the expected schema."""


def _extract_json(raw: str) -> str:
    """Strip markdown fences and surrounding whitespace from a raw LLM response."""
    stripped = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence:
        return fence.group(1)
    return stripped


class LLMNERAgent(BaseAgent[PipelineState]):
    """LLM-backed BIO tagger for the full_agentic NER pipeline.

    Parameters
    ----------
    llm_client:
        Any ``LLMClient`` implementation. Pass ``MockLLMClient`` in tests.
    output_slot:
        Which specialist slot to write to — ``"contextual"`` (default),
        ``"lexical"``, or ``"logic"``. Determines which ``NERConsensusAgent``
        vote this agent occupies.
    name:
        Optional agent name for logging and ``AgentOutput.agent_name``.
    logger:
        Optional pre-configured logger.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        output_slot: str = "contextual",
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name=name or "LLMNERAgent", logger=logger)
        if output_slot not in _SLOT_ATTR:
            raise ValueError(
                f"output_slot must be one of {sorted(_SLOT_ATTR)}, got '{output_slot}'."
            )
        self.llm_client = llm_client
        self.output_slot = output_slot

    # ------------------------------------------------------------------
    # Validation hooks
    # ------------------------------------------------------------------

    def validate_before(self, state: PipelineState) -> None:
        if not state.task_config.labels:
            raise ValueError("LLMNERAgent: state.task_config.labels is empty.")

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """Call the LLM, parse per-token tags, write result to the chosen slot."""
        if state.task_config.task_type != "sequence_labeling":
            self.logger.debug(_SKIP_NOTE)
            state.append_history(component=self.name, summary=_SKIP_NOTE)
            return state

        task = state.task_config
        tokens: List[str] = state.extras.get("tokens") or state.input_text.split()

        prompt = build_user_prompt(
            task_name=task.task_name,
            labels=task.labels,
            tokens=tokens,
            text=state.input_text,
        )
        # Entity types + descriptions are derived from the task config (dynamic,
        # like the topic classifier's categories) — not hardcoded in the agent.
        system_prompt = get_system_prompt(
            labels=task.labels,
            label_descriptions=task.label_descriptions,
        )
        self.logger.debug("%s: sending prompt (%d tokens)", self.name, len(tokens))
        raw_response = self.llm_client.generate(f"{system_prompt}\n\n{prompt}")

        try:
            tags, reasoning = self._parse_response(raw_response, len(tokens))
            note = ""
        except LLMNERParseError as exc:
            self.logger.warning("%s: parse error — %s", self.name, exc)
            tags = ["O"] * len(tokens)
            reasoning = ""
            note = f"{_PARSE_FAIL_NOTE} ({exc})"

        # Align to token count and validate tags against the task label set.
        tags, len_note = self._align_length(tags, len(tokens))
        valid = set(task.labels)
        norm_tags = [coerce_to_valid(t, valid) for t in tags]
        note = " ".join(n for n in (note, len_note) if n).strip()

        seq_out = SequenceLabelingOutput(
            tags=[TokenTag(token=tok, tag=tag, confidence=_LLM_CONFIDENCE)
                  for tok, tag in zip(tokens, norm_tags)],
            notes=note or reasoning,
            features={"raw_llm_response": raw_response, "reasoning": reasoning},
        )
        setattr(state, _SLOT_ATTR[self.output_slot], AgentOutput(
            agent_name=self.name,
            model_output=ModelOutput(),   # intentionally empty for NER
            sequence_output=seq_out,
            notes=seq_out.notes,
        ))

        tag_summary = [f"{tt.token}:{tt.tag}" for tt in seq_out.tags]
        state.append_history(
            component=self.name,
            summary=f"LLM NER tagging complete ({self.output_slot} slot): "
                    f"{len(seq_out.tags)} token(s). {reasoning}".strip(),
            outputs={"tags": tag_summary, "fallback": bool(note), "reasoning": reasoning},
        )
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, n_tokens: int) -> tuple[List[str], str]:
        """Parse the LLM JSON response into (tags, reasoning).

        Raises
        ------
        LLMNERParseError
            When the response is not valid JSON or is missing the "tags" list.
        """
        try:
            data = json.loads(_extract_json(raw))
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMNERParseError(f"JSON decode failed: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMNERParseError(f"Expected a JSON object, got {type(data).__name__}")
        if "tags" not in data or not isinstance(data["tags"], list):
            raise LLMNERParseError("Missing or non-list 'tags'.")
        tags = [str(t) for t in data["tags"]]
        reasoning = str(data.get("reasoning", ""))
        return tags, reasoning

    @staticmethod
    def _align_length(tags: List[str], n_tokens: int) -> tuple[List[str], str]:
        """Pad with 'O' or truncate so len(tags) == n_tokens; note if changed."""
        if len(tags) == n_tokens:
            return tags, ""
        if len(tags) < n_tokens:
            return tags + ["O"] * (n_tokens - len(tags)), _LEN_MISMATCH_NOTE
        return tags[:n_tokens], _LEN_MISMATCH_NOTE
