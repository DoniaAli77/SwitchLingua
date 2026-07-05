"""sequential_sentiment_v2 — forward-pragmatics staged pipeline (opt-in).

Redesign of v1 that removes the confirmation-anchored review stage. Pragmatics is
extracted **upstream as structured features**, and polarity is decided **once**,
feature-aware, with no prior label shown to the final stage:

    text -> SeqV2IntentAgent            (Stage 1, lean opinion-existence)
         -> SeqV2PragmaticFeaturesAgent (Stage 2, structured features; NO label)
         -> SeqV2PolarityResolverAgent  (Stage 3, decides label once, feature-aware)
         -> SequentialControllerV2       (Stage 4, deterministic; no keep/revise)

Reuses the v1 stage machinery (`_SeqStageBase`: LLM call + one retry + safe coerce +
persistence under ``state.extras[SEQ_KEY]``). The default pipeline never constructs any
of these. See EXPERIMENT_SEQUENTIAL_SENTIMENT_V2_FORWARD_PRAGMATICS_DESIGN.md.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.agents.base_agent import BaseAgent
from src.agents.sequential_sentiment import (
    SEQ_KEY,
    TAU_INTENT_DEFAULT,
    TAU_LOW_DEFAULT,
    _SeqStageBase,
    _coerce_confidence,
    _coerce_opinion,
    _coerce_str_list,
    _neutral_or_first,
    _seq_store,
)
from src.prompts.sequential_sentiment_v2_prompts import (
    INTENT_V2_SYSTEM_PROMPT,
    PRAGMATIC_FEATURES_SYSTEM_PROMPT,
    POLARITY_RESOLVER_SYSTEM_PROMPT,
    build_intent_v2_user_prompt,
    build_pragmatic_features_user_prompt,
    build_polarity_resolver_user_prompt,
)
from src.state.schema import ConsensusOutput, FinalOutput, PipelineState

log = logging.getLogger(__name__)

_SPEECH_ACTS = frozenset({"evaluate", "describe", "ask", "advise", "quote", "other"})
_USE_MENTION = frozenset({"use", "mention", "platform_meta"})
_ATTRIBUTION = frozenset({"author", "other", "none"})
_DESC_EVAL = frozenset({"evaluation", "description", "mixed"})
_IMPLICIT = frozenset({"positive", "negative", "none"})
_STRENGTH = frozenset({"none", "weak", "moderate", "strong"})


# ---------------------------------------------------------------------------
# Stage 1 — Intent (lean)
# ---------------------------------------------------------------------------

_INTENT_V2_REQUIRED = frozenset({"opinion_expressed", "target", "confidence", "evidence"})


class SeqV2IntentAgent(_SeqStageBase):
    """Stage 1: lean opinion-existence gate (no sentiment label)."""

    stage_key = "intent"

    def _build_prompt(self, state: PipelineState, store: Dict[str, Any]) -> str:
        return f"{INTENT_V2_SYSTEM_PROMPT}\n\n{build_intent_v2_user_prompt(state.input_text)}"

    def _normalize(self, data: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        missing = _INTENT_V2_REQUIRED - data.keys()
        if missing:
            raise ValueError(f"missing keys: {sorted(missing)}")
        target = data["target"]
        return {
            "opinion_expressed": _coerce_opinion(data["opinion_expressed"]),
            "target": None if target is None else str(target),
            "confidence": _coerce_confidence(data["confidence"]),
            "evidence": _coerce_str_list(data["evidence"]),
        }

    def _safe_default(self, state: PipelineState) -> Dict[str, Any]:
        return {
            "opinion_expressed": "unclear",
            "target": None,
            "confidence": 0.0,
            "evidence": [],
            "coerced": True,
        }


# ---------------------------------------------------------------------------
# Stage 2 — Pragmatic feature extractor (NO sentiment label)
# ---------------------------------------------------------------------------

_PRAGMATIC_FEATURES_REQUIRED = frozenset({
    "speech_act", "target", "target_attribution", "use_vs_mention", "platform_meta",
    "description_vs_evaluation", "sarcasm_or_irony", "implicit_stance", "stance_strength",
    "confidence", "evidence",
})


def _pick(value: Any, allowed: frozenset, default: str) -> str:
    v = str(value).strip().lower()
    return v if v in allowed else default


class SeqV2PragmaticFeaturesAgent(_SeqStageBase):
    """Stage 2: structured pragmatic features conditioned on Stage-1 intent. No label."""

    stage_key = "pragmatic"

    def _build_prompt(self, state: PipelineState, store: Dict[str, Any]) -> str:
        intent = store.get("intent", {})
        user = build_pragmatic_features_user_prompt(state.input_text, intent)
        return f"{PRAGMATIC_FEATURES_SYSTEM_PROMPT}\n\n{user}"

    def _normalize(self, data: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        missing = _PRAGMATIC_FEATURES_REQUIRED - data.keys()
        if missing:
            raise ValueError(f"missing keys: {sorted(missing)}")
        target = data["target"]
        return {
            "speech_act": _pick(data["speech_act"], _SPEECH_ACTS, "other"),
            "target": None if target is None else str(target),
            "target_attribution": _pick(data["target_attribution"], _ATTRIBUTION, "none"),
            "use_vs_mention": _pick(data["use_vs_mention"], _USE_MENTION, "use"),
            "platform_meta": bool(data["platform_meta"]),
            "description_vs_evaluation": _pick(data["description_vs_evaluation"], _DESC_EVAL, "mixed"),
            "sarcasm_or_irony": bool(data["sarcasm_or_irony"]),
            "implicit_stance": _pick(data["implicit_stance"], _IMPLICIT, "none"),
            "stance_strength": _pick(data["stance_strength"], _STRENGTH, "none"),
            "confidence": _coerce_confidence(data["confidence"]),
            "evidence": _coerce_str_list(data["evidence"]),
        }

    def _safe_default(self, state: PipelineState) -> Dict[str, Any]:
        # Neutral-shaped features that do NOT force the no-opinion gate (use_vs_mention="use"
        # + description="mixed" fail the gate's condition), so a broken feature stage defers
        # the decision to the polarity resolver rather than forcing neutral.
        return {
            "speech_act": "other", "target": None, "target_attribution": "none",
            "use_vs_mention": "use", "platform_meta": False,
            "description_vs_evaluation": "mixed", "sarcasm_or_irony": False,
            "implicit_stance": "none", "stance_strength": "none",
            "confidence": 0.0, "evidence": [], "coerced": True,
        }


# ---------------------------------------------------------------------------
# Stage 3 — Polarity resolver (decides once, feature-aware, no prior label shown)
# ---------------------------------------------------------------------------

_POLARITY_RESOLVER_REQUIRED = frozenset({"label", "confidence", "reasoning", "evidence"})


class SeqV2PolarityResolverAgent(_SeqStageBase):
    """Stage 3: final polarity decision using text + intent + pragmatic features."""

    stage_key = "polarity"

    def _build_prompt(self, state: PipelineState, store: Dict[str, Any]) -> str:
        task = state.task_config
        user = build_polarity_resolver_user_prompt(
            task_name=task.task_name,
            labels=task.labels,
            label_descriptions=task.label_descriptions,
            text=state.input_text,
            intent=store.get("intent", {}),
            pragmatic=store.get("pragmatic", {}),
        )
        return f"{POLARITY_RESOLVER_SYSTEM_PROMPT}\n\n{user}"

    def _normalize(self, data: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        missing = _POLARITY_RESOLVER_REQUIRED - data.keys()
        if missing:
            raise ValueError(f"missing keys: {sorted(missing)}")
        label = str(data["label"]).strip()
        if not state.task_config.is_allowed_label(label):
            raise ValueError(f"invalid label {label!r}")
        return {
            "label": label,
            "confidence": _coerce_confidence(data["confidence"]),
            "used_features": _coerce_str_list(data.get("used_features", [])),
            "reasoning": str(data["reasoning"]),
            "evidence": _coerce_str_list(data["evidence"]),
        }

    def _safe_default(self, state: PipelineState) -> Dict[str, Any]:
        return {
            "label": _neutral_or_first(state),
            "confidence": 0.0,
            "used_features": [],
            "reasoning": "coerced default (polarity parse failed)",
            "evidence": [],
            "coerced": True,
        }


# ---------------------------------------------------------------------------
# Stage 4 — Deterministic controller (no keep/revise)
# ---------------------------------------------------------------------------

class SequentialControllerV2(BaseAgent[PipelineState]):
    """Deterministic v2 controller. Rules (first match wins):

    1. **No-opinion neutral** — intent says no opinion (conf >= TAU_INTENT) AND the
       pragmatic features agree there is no evaluation (implicit_stance none, and either
       use_vs_mention != "use" or description_vs_evaluation == "description"). Cross-checked
       gate: no single stage can force neutral. `decided_by=intent_no_opinion`.
    2. **Feature-aware polarity** — polarity.confidence >= TAU_LOW → Stage-3 label.
       `decided_by=polarity_feature_aware`.
    3. **Weak / conflicted fallback** — else primary (if USE_PRIMARY_FALLBACK and usable)
       → `fallback_primary`; otherwise the Stage-3 label → `fallback_polarity`.

    There is no keep/revise / pragmatic-revision rule — sarcasm/mention are resolved inside
    Stage 3. Primary is router + safe fallback only, never a voter.
    """

    def __init__(
        self,
        tau_intent: float = TAU_INTENT_DEFAULT,
        tau_low: float = TAU_LOW_DEFAULT,
        use_primary_fallback: bool = True,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name=name or "SequentialControllerV2", logger=logger)
        self.tau_intent = tau_intent
        self.tau_low = tau_low
        self.use_primary_fallback = use_primary_fallback

    def validate_before(self, state: PipelineState) -> None:
        if not state.task_config.labels:
            raise ValueError("SequentialControllerV2: state.task_config.labels is empty.")

    def run(self, state: PipelineState) -> PipelineState:
        store = _seq_store(state)
        intent = store.get("intent", {}) or {}
        pragmatic = store.get("pragmatic", {}) or {}
        polarity = store.get("polarity", {}) or {}

        label, decided_by, fallback_path = self._decide(state, intent, pragmatic, polarity)

        if not state.task_config.is_allowed_label(label):
            label = _neutral_or_first(state)
            fallback_path = "coerce_invalid_label"

        confidence = self._confidence_for(decided_by, polarity, state)

        store["thresholds"] = {
            "tau_intent": self.tau_intent,
            "tau_low": self.tau_low,
            "use_primary_fallback": self.use_primary_fallback,
        }
        store["decided_by"] = decided_by
        store["fallback_path"] = fallback_path
        store["final_label"] = label

        rationale = (
            f"sequential_sentiment_v2: decided_by={decided_by}"
            + (f", fallback_path={fallback_path}" if fallback_path else "")
        )
        state.consensus_output = ConsensusOutput(
            label=label, confidence=confidence, votes={}, rationale=rationale
        )
        state.final_output = FinalOutput(
            label=label,
            confidence=confidence,
            payload={
                "source": "sequential_sentiment_v2",
                "decided_by": decided_by,
                "fallback_path": fallback_path,
                "used_features": polarity.get("used_features", []),
            },
        )

        self.logger.debug(
            "%s: label=%s decided_by=%s fallback=%s", self.name, label, decided_by, fallback_path
        )
        state.append_history(
            component=self.name,
            summary=f"Sequential-v2 decision: '{label}' via {decided_by}.",
            outputs={
                "label": label,
                "confidence": confidence,
                "decided_by": decided_by,
                "fallback_path": fallback_path,
                "intent_opinion": intent.get("opinion_expressed"),
                "sarcasm_or_irony": pragmatic.get("sarcasm_or_irony"),
                "use_vs_mention": pragmatic.get("use_vs_mention"),
                "implicit_stance": pragmatic.get("implicit_stance"),
                "polarity_label": polarity.get("label"),
                "used_features": polarity.get("used_features", []),
            },
        )
        return state

    def _decide(self, state, intent, pragmatic, polarity):
        neutral = _neutral_or_first(state)
        opinion = intent.get("opinion_expressed")
        intent_conf = _coerce_confidence(intent.get("confidence", 0.0))
        implicit = pragmatic.get("implicit_stance", "none")
        uvm = pragmatic.get("use_vs_mention", "use")
        desc = pragmatic.get("description_vs_evaluation", "mixed")
        pol_label = polarity.get("label")
        pol_conf = _coerce_confidence(polarity.get("confidence", 0.0))

        # Rule 1 — cross-checked no-opinion neutral gate.
        if (
            opinion is False
            and intent_conf >= self.tau_intent
            and implicit == "none"
            and (uvm != "use" or desc == "description")
        ):
            return neutral, "intent_no_opinion", None

        # Rule 2 — trust the feature-aware polarity.
        if pol_label is not None and pol_conf >= self.tau_low:
            return pol_label, "polarity_feature_aware", None

        # Rule 3 — weak / conflicted fallback.
        primary = state.primary_model_output
        primary_ok = (
            primary is not None
            and primary.label is not None
            and state.task_config.is_allowed_label(primary.label)
        )
        if self.use_primary_fallback and primary_ok:
            return primary.label, "fallback_primary", "weak_polarity"
        if pol_label is not None:
            return pol_label, "fallback_polarity", "weak_polarity"
        if primary_ok:
            return primary.label, "fallback_primary", "no_polarity"
        return neutral, "fallback_neutral", "no_polarity_no_primary"

    def _confidence_for(self, decided_by, polarity, state):
        if decided_by in ("polarity_feature_aware", "fallback_polarity"):
            return _coerce_confidence(polarity.get("confidence", 0.0))
        if decided_by == "fallback_primary":
            p = state.primary_model_output
            return p.confidence if p is not None and p.confidence is not None else 0.0
        if decided_by == "intent_no_opinion":
            return _coerce_confidence(polarity.get("confidence", 0.0))
        return 0.0
