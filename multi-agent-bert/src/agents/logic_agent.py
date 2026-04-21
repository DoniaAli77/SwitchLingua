"""Logic agent for stateful multi-agent text classification pipelines.

Scores each configured label by testing compiled regex rules against the input
text.  Each rule that matches contributes one vote to its label.

Rule format
-----------
``rule_map`` is ``{label: [pattern_string, …]}``.  Every pattern is compiled
with ``re.IGNORECASE`` and ``re.UNICODE`` at construction time so matching is
fast and Arabic / Unicode characters are handled correctly.  Patterns that fail
to compile are skipped with a warning rather than crashing the pipeline.

No-match fallback
-----------------
When no rule fires across all labels the agent returns a uniform probability
distribution (``1 / |labels|``) and picks ``labels[0]`` as the fallback,
giving it an explicit low confidence rather than ``None``.

Tiebreak
--------
When multiple labels share the top vote count the first label in
``task_config.labels`` with that count wins, making results deterministic.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, NamedTuple, Optional

from src.agents.base_agent import BaseAgent
from src.state.schema import AgentOutput, ModelOutput, PipelineState

_NO_MATCH_NOTE = "No rules triggered; uniform fallback applied."


class _CompiledRule(NamedTuple):
    label: str
    pattern_str: str
    regex: re.Pattern[str]


class LogicAgent(BaseAgent[PipelineState]):
    """Regex-rule label scorer for code-switched text classification.

    Parameters
    ----------
    rule_map:
        Mapping of ``{label: [regex_pattern, …]}``.  Patterns are compiled
        once at construction.  Labels absent from the map get no rules
        (score = 0).  Pass an empty dict to get pure fallback behaviour.
    name:
        Optional agent name used for logging and ``AgentOutput.agent_name``.
    logger:
        Optional pre-configured logger; a default one is created otherwise.
    """

    def __init__(
        self,
        rule_map: Optional[Dict[str, List[str]]] = None,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name=name or "LogicAgent", logger=logger)
        self._compiled: List[_CompiledRule] = []
        for label, patterns in (rule_map or {}).items():
            for pat in patterns:
                try:
                    self._compiled.append(
                        _CompiledRule(
                            label=label,
                            pattern_str=pat,
                            regex=re.compile(pat, re.IGNORECASE | re.UNICODE),
                        )
                    )
                except re.error as exc:
                    logging.getLogger(self.name).warning(
                        "Skipping invalid regex pattern '%s' for label '%s': %s",
                        pat, label, exc,
                    )

    # ------------------------------------------------------------------
    # Validation hooks
    # ------------------------------------------------------------------

    def validate_before(self, state: PipelineState) -> None:
        if not state.input_text or not state.input_text.strip():
            raise ValueError("LogicAgent: state.input_text is empty or blank.")
        if not state.task_config.labels:
            raise ValueError("LogicAgent: state.task_config.labels is empty.")

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """Test each rule against input text, score labels, write to state.logic_output."""

        text = state.input_text
        labels = state.task_config.labels

        raw_scores: Dict[str, int] = {lbl: 0 for lbl in labels}
        triggered: List[str] = []          # "label:pattern" strings

        for rule in self._compiled:
            if rule.label not in raw_scores:
                # Rule targets a label not in task_config — skip silently.
                continue
            if rule.regex.search(text):
                raw_scores[rule.label] += 1
                triggered.append(f"{rule.label}:{rule.pattern_str}")

        total_hits = sum(raw_scores.values())

        if total_hits == 0:
            uniform = round(1.0 / len(labels), 6)
            probabilities = {lbl: uniform for lbl in labels}
            best_label = labels[0]
            reasoning = _NO_MATCH_NOTE
        else:
            probabilities = {
                lbl: round(raw_scores[lbl] / total_hits, 6) for lbl in labels
            }
            best_label = max(labels, key=lambda lbl: raw_scores[lbl])
            fired_count = sum(1 for s in raw_scores.values() if s > 0)
            reasoning = (
                f"Triggered {total_hits} rule(s) across {fired_count} label(s). "
                f"Best: '{best_label}' ({probabilities[best_label]:.2%})."
            )

        model_output = ModelOutput(
            label=best_label,
            confidence=probabilities[best_label],
            probabilities=probabilities,
            raw_text=text,
        )

        state.logic_output = AgentOutput(
            agent_name=self.name,
            model_output=model_output,
            notes=reasoning,
            features={
                "triggered_rules": triggered,
                "raw_scores": raw_scores,
            },
        )

        self.logger.debug(
            "%s: label=%s confidence=%.4f triggered=%s",
            self.name,
            best_label,
            probabilities[best_label],
            triggered,
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
                "triggered_rules": triggered,
                "raw_scores": dict(raw_scores),
            },
        )
        return state
