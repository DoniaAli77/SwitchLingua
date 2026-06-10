"""Lexical agent for stateful multi-agent text classification pipelines.

Scores each configured label by counting keyword matches in the input text,
then normalises the raw counts into a probability distribution.  Supports
both Arabic (right-to-left, non-ASCII) and Latin-script (English, French, …)
keyword lists in the same map.

Keyword matching rules
----------------------
* **Latin-script keywords** (all ASCII): whole-word, case-insensitive regex match
  (``\\bkeyword\\b``), so "AI" does not match "FAIL".
* **Arabic / non-ASCII keywords and multi-word phrases**: substring match inside
  the original text so that connected-script words are found correctly.

No-match fallback
-----------------
When zero keywords match across all labels the agent returns a uniform
probability distribution (``1 / |labels|``) and picks the first label from
``task_config.labels`` as the best guess, giving it explicit low confidence
rather than ``None``.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from src.agents._abstain import abstain_output
from src.agents.base_agent import BaseAgent
from src.state.schema import AgentOutput, ModelOutput, PipelineState

# Low-confidence sentinel used when no keywords matched at all.
_NO_MATCH_NOTE = "No keywords matched; uniform fallback applied."


def _keyword_in_text(keyword: str, text: str) -> bool:
    """Return True if *keyword* appears in *text*.

    Uses whole-word regex for pure-ASCII keywords and plain substring search
    for Arabic / mixed-script / multi-word keywords.
    """
    if keyword.isascii():
        pattern = r"(?i)\b" + re.escape(keyword) + r"\b"
        return bool(re.search(pattern, text))
    # Non-ASCII (Arabic, Devanagari, …) or multi-word phrase: substring search.
    return keyword in text


class LexicalAgent(BaseAgent[PipelineState]):
    """Keyword-based label scorer for code-switched text classification.

    Parameters
    ----------
    keyword_map:
        Mapping of ``{label: [keyword, …]}``.  Labels missing from the map
        are treated as having no keywords (score = 0).  Pass an empty dict
        to get pure fallback behaviour.
    name:
        Optional agent name used for logging and ``AgentOutput.agent_name``.
    logger:
        Optional pre-configured logger; a default one is created otherwise.
    """

    def __init__(
        self,
        keyword_map: Optional[Dict[str, List[str]]] = None,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name=name or "LexicalAgent", logger=logger)
        self.keyword_map: Dict[str, List[str]] = keyword_map or {}

    # ------------------------------------------------------------------
    # Validation hooks
    # ------------------------------------------------------------------

    def validate_before(self, state: PipelineState) -> None:
        if not state.input_text or not state.input_text.strip():
            raise ValueError("LexicalAgent: state.input_text is empty or blank.")
        if not state.task_config.labels:
            raise ValueError("LexicalAgent: state.task_config.labels is empty.")

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """Score labels by keyword frequency, write result to state.lexical_output."""

        text = state.input_text
        labels = state.task_config.labels

        # --- count keyword hits per label ---
        raw_scores: Dict[str, int] = {}
        evidence_by_label: Dict[str, List[str]] = {}

        for label in labels:
            keywords = self.keyword_map.get(label, [])
            matched = [kw for kw in keywords if _keyword_in_text(kw, text)]
            raw_scores[label] = len(matched)
            evidence_by_label[label] = matched

        total_hits = sum(raw_scores.values())

        # --- normalise into probabilities ---
        if total_hits == 0:
            # No keyword signal → abstain (no vote) rather than defaulting to a label.
            state.lexical_output = abstain_output(self.name, state, _NO_MATCH_NOTE)
            state.append_history(
                component=self.name,
                summary=f"No keyword match — abstaining (no vote). {_NO_MATCH_NOTE}",
                outputs={"abstained": True, "raw_scores": dict(raw_scores)},
            )
            return state

        probabilities = {
            lbl: round(raw_scores[lbl] / total_hits, 6) for lbl in labels
        }
        # Stable tiebreak: first label with max score wins (deterministic).
        best_label = max(labels, key=lambda lbl: raw_scores[lbl])
        all_evidence = [
            f"{lbl}:{kw}"
            for lbl in labels
            for kw in evidence_by_label[lbl]
        ]
        matched_label_count = sum(1 for s in raw_scores.values() if s > 0)
        reasoning = (
            f"Matched {total_hits} keyword(s) across "
            f"{matched_label_count} label(s). "
            f"Best: '{best_label}' ({probabilities[best_label]:.2%})."
        )

        model_output = ModelOutput(
            label=best_label,
            confidence=probabilities[best_label],
            probabilities=probabilities,
            raw_text=text,
        )

        state.lexical_output = AgentOutput(
            agent_name=self.name,
            model_output=model_output,
            notes=reasoning,
            features={
                "matched_evidence": all_evidence,
                "raw_scores": raw_scores,
            },
        )

        self.logger.debug(
            "%s: label=%s confidence=%.4f evidence=%s",
            self.name,
            best_label,
            probabilities[best_label],
            all_evidence,
        )

        state.append_history(
            component=self.name,
            summary=(
                f"Label '{best_label}' (confidence={probabilities[best_label]:.3f}). "
                f"{reasoning}"
            ),
            outputs={
                "label": best_label,
                "confidence": probabilities[best_label],
                "probabilities": dict(probabilities),
                "evidence": all_evidence,
                "raw_scores": dict(raw_scores),
            },
        )
        return state
