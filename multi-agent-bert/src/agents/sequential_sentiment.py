"""Sequential (staged-reasoning) sentiment pipeline — ``sequential_sentiment_v1``.

Experimental, **opt-in** alternative to the parallel voting pipeline. Selected via
the ``sequential_sentiment_v1`` sentiment agent variant; the default pipeline never
constructs any of these agents, so shipped behaviour is unchanged.

Architecture (3 LLM stages + 1 deterministic controller)::

    text -> SeqIntentAgent  (Stage 1)  -> intent JSON
         -> SeqPolarityAgent (Stage 2)  -> polarity JSON   (conditioned on intent)
         -> SeqPragmaticAgent(Stage 3)  -> pragmatic JSON  (conditioned on intent+polarity)
         -> SequentialController (Stage 4, no LLM) -> final_output + consensus_output

All intermediate stage outputs, per-stage confidences, retry/coercion events, the
chosen ``decided_by`` rule and any ``fallback_path`` are persisted under
``state.extras[SEQ_KEY]`` (JSON-serializable; no schema change required).

Error handling (per the approved design):
* one retry per stage on malformed JSON (same temperature-0 client);
* on a second failure, a safe per-stage default is coerced and logged — no crash;
* controller fallback order is Polarity -> Primary -> neutral.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.llm.base_client import LLMClient, LLMClientError
from src.prompts.sequential_sentiment_prompts import (
    INTENT_SYSTEM_PROMPT,
    POLARITY_SYSTEM_PROMPT,
    PRAGMATIC_SYSTEM_PROMPT,
    build_intent_user_prompt,
    build_polarity_user_prompt,
    build_pragmatic_user_prompt,
)
from src.state.schema import ConsensusOutput, FinalOutput, PipelineState

#: Key under ``state.extras`` where the whole sequential trace is stored.
SEQ_KEY = "sequential_sentiment"

#: Default controller thresholds (config-exposed via the controller constructor).
TAU_INTENT_DEFAULT = 0.60
TAU_REVISE_DEFAULT = 0.60
TAU_LOW_DEFAULT = 0.45

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> str:
    """Strip markdown fences / surrounding whitespace from a raw LLM response."""
    stripped = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence:
        return fence.group(1)
    return stripped


def _coerce_confidence(value: Any) -> float:
    """Coerce a confidence value to a float clamped to [0, 1]; 0.0 on failure."""
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.0
    if c < 0.0:
        return 0.0
    if c > 1.0:
        return 1.0
    return c


def _coerce_str_list(value: Any) -> List[str]:
    """Coerce an evidence field to a list of strings (empty list on failure)."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None:
        return []
    return [str(value)]


def _seq_store(state: PipelineState) -> Dict[str, Any]:
    """Return (creating if needed) the sequential trace dict on ``state.extras``."""
    store = state.extras.get(SEQ_KEY)
    if not isinstance(store, dict):
        store = {"stage_events": []}
        state.extras[SEQ_KEY] = store
    store.setdefault("stage_events", [])
    return store


def _record_event(store: Dict[str, Any], stage: str, event: str, detail: str = "") -> None:
    store["stage_events"].append({"stage": stage, "event": event, "detail": detail})


class _SeqStageBase(BaseAgent[PipelineState]):
    """Common LLM-call + one-retry + safe-parse machinery for the three stages.

    Subclasses provide ``stage_key``, a prompt builder, a required-key set, a
    parser/normalizer, and a safe default. The base class handles: building the
    full prompt, calling the client (once, then one retry on parse failure),
    coercing to a safe default on repeated failure, persisting the result under
    ``state.extras[SEQ_KEY][stage_key]``, and history.
    """

    stage_key: str = "stage"

    def __init__(
        self,
        llm_client: LLMClient,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name=name or self.__class__.__name__, logger=logger)
        self.llm_client = llm_client

    # -- hooks subclasses implement ------------------------------------------

    def _build_prompt(self, state: PipelineState, store: Dict[str, Any]) -> str:
        raise NotImplementedError

    def _normalize(self, data: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        """Validate + coerce a parsed dict into the canonical stage record."""
        raise NotImplementedError

    def _safe_default(self, state: PipelineState) -> Dict[str, Any]:
        """Return the safe fallback record used when parsing fails twice."""
        raise NotImplementedError

    # -- validation ----------------------------------------------------------

    def validate_before(self, state: PipelineState) -> None:
        if not state.input_text or not state.input_text.strip():
            raise ValueError(f"{self.name}: state.input_text is empty or blank.")
        if not state.task_config.labels:
            raise ValueError(f"{self.name}: state.task_config.labels is empty.")

    # -- core ----------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        store = _seq_store(state)
        prompt = self._build_prompt(state, store)

        record, raw, event = self._call_with_retry(prompt, state, store)
        record["_raw"] = raw
        store[self.stage_key] = record

        self.logger.debug("%s: %s -> %s", self.name, self.stage_key, event)
        state.append_history(
            component=self.name,
            summary=f"{self.stage_key} stage: {event}.",
            outputs={k: v for k, v in record.items() if k != "_raw"},
        )
        return state

    def _call_with_retry(
        self, prompt: str, state: PipelineState, store: Dict[str, Any]
    ):
        """Call the client, retry once on parse failure, else coerce a default."""
        last_raw = ""
        for attempt in (1, 2):
            try:
                raw = self.llm_client.generate(prompt)
            except LLMClientError as exc:
                _record_event(store, self.stage_key, "llm_error", f"attempt {attempt}: {exc}")
                last_raw = f"<llm_error: {exc}>"
                continue
            last_raw = raw
            try:
                data = json.loads(_extract_json(raw))
                if not isinstance(data, dict):
                    raise ValueError(f"expected object, got {type(data).__name__}")
                record = self._normalize(data, state)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
                _record_event(
                    store, self.stage_key,
                    "retry" if attempt == 1 else "coerced_default",
                    f"attempt {attempt}: {exc}",
                )
                continue
            if attempt == 2:
                _record_event(store, self.stage_key, "parse_ok_after_retry", "")
            return record, last_raw, "parse_ok" if attempt == 1 else "parse_ok_after_retry"

        # Both attempts failed → safe default.
        return self._safe_default(state), last_raw, "coerced_default"


# ---------------------------------------------------------------------------
# Stage 1 — Intent / opinion-expression detector
# ---------------------------------------------------------------------------

_INTENT_REQUIRED = frozenset(
    {"opinion_expressed", "target", "speech_act", "use_vs_mention", "confidence", "evidence"}
)
_SPEECH_ACTS = frozenset({"evaluate", "describe", "ask", "advise", "quote", "other"})
_USE_MENTION = frozenset({"use", "mention", "platform_meta"})


def _coerce_opinion(value: Any) -> Any:
    """Coerce opinion_expressed to True / False / 'unclear'."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes"):
            return True
        if low in ("false", "no"):
            return False
    return "unclear"


class SeqIntentAgent(_SeqStageBase):
    """Stage 1: decide whether the author expresses an opinion (no sentiment label)."""

    stage_key = "intent"

    def _build_prompt(self, state: PipelineState, store: Dict[str, Any]) -> str:
        user = build_intent_user_prompt(state.input_text)
        return f"{INTENT_SYSTEM_PROMPT}\n\n{user}"

    def _normalize(self, data: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        missing = _INTENT_REQUIRED - data.keys()
        if missing:
            raise ValueError(f"missing keys: {sorted(missing)}")
        speech_act = str(data["speech_act"]).strip().lower()
        use_vs_mention = str(data["use_vs_mention"]).strip().lower()
        target = data["target"]
        return {
            "opinion_expressed": _coerce_opinion(data["opinion_expressed"]),
            "target": None if target is None else str(target),
            "speech_act": speech_act if speech_act in _SPEECH_ACTS else "other",
            "use_vs_mention": use_vs_mention if use_vs_mention in _USE_MENTION else "use",
            "confidence": _coerce_confidence(data["confidence"]),
            "evidence": _coerce_str_list(data["evidence"]),
        }

    def _safe_default(self, state: PipelineState) -> Dict[str, Any]:
        # Neutral default: "unclear" with confidence 0.0 disables the controller's
        # high-confidence no-opinion branch, so a broken intent stage cannot
        # unconditionally force neutral (cascade guard).
        return {
            "opinion_expressed": "unclear",
            "target": None,
            "speech_act": "other",
            "use_vs_mention": "use",
            "confidence": 0.0,
            "evidence": [],
            "coerced": True,
        }


# ---------------------------------------------------------------------------
# Stage 2 — Polarity resolver
# ---------------------------------------------------------------------------

_POLARITY_REQUIRED = frozenset({"label", "confidence", "reasoning", "evidence"})


class SeqPolarityAgent(_SeqStageBase):
    """Stage 2: assign a sentiment label conditioned on the Stage-1 intent JSON."""

    stage_key = "polarity"

    def _build_prompt(self, state: PipelineState, store: Dict[str, Any]) -> str:
        task = state.task_config
        intent = store.get("intent", {})
        user = build_polarity_user_prompt(
            task_name=task.task_name,
            labels=task.labels,
            label_descriptions=task.label_descriptions,
            text=state.input_text,
            intent=intent,
        )
        return f"{POLARITY_SYSTEM_PROMPT}\n\n{user}"

    def _normalize(self, data: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        missing = _POLARITY_REQUIRED - data.keys()
        if missing:
            raise ValueError(f"missing keys: {sorted(missing)}")
        label = str(data["label"]).strip()
        if not state.task_config.is_allowed_label(label):
            raise ValueError(f"invalid label {label!r}")
        return {
            "label": label,
            "confidence": _coerce_confidence(data["confidence"]),
            "mixed": bool(data.get("mixed", False)),
            "reasoning": str(data["reasoning"]),
            "evidence": _coerce_str_list(data["evidence"]),
        }

    def _safe_default(self, state: PipelineState) -> Dict[str, Any]:
        # Zero-confidence polarity → forces the controller's weak/fallback branch,
        # which defers to the primary (or Polarity if the primary is unusable).
        return {
            "label": _neutral_or_first(state),
            "confidence": 0.0,
            "mixed": False,
            "reasoning": "coerced default (polarity parse failed)",
            "evidence": [],
            "coerced": True,
        }


# ---------------------------------------------------------------------------
# Stage 3 — Pragmatic verifier
# ---------------------------------------------------------------------------

_PRAGMATIC_REQUIRED = frozenset({"keep_or_revise", "final_label", "confidence", "reasoning", "evidence"})


class SeqPragmaticAgent(_SeqStageBase):
    """Stage 3: keep or revise the polarity based on sarcasm / implicature / mention."""

    stage_key = "pragmatic"

    def _build_prompt(self, state: PipelineState, store: Dict[str, Any]) -> str:
        task = state.task_config
        intent = store.get("intent", {})
        polarity = store.get("polarity", {})
        user = build_pragmatic_user_prompt(
            task_name=task.task_name,
            labels=task.labels,
            label_descriptions=task.label_descriptions,
            text=state.input_text,
            intent=intent,
            polarity=polarity,
        )
        return f"{PRAGMATIC_SYSTEM_PROMPT}\n\n{user}"

    def _normalize(self, data: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        missing = _PRAGMATIC_REQUIRED - data.keys()
        if missing:
            raise ValueError(f"missing keys: {sorted(missing)}")
        decision = str(data["keep_or_revise"]).strip().lower()
        if decision not in ("keep", "revise"):
            raise ValueError(f"invalid keep_or_revise {decision!r}")
        final_label = str(data["final_label"]).strip()
        if not state.task_config.is_allowed_label(final_label):
            raise ValueError(f"invalid final_label {final_label!r}")
        return {
            "keep_or_revise": decision,
            "final_label": final_label,
            "confidence": _coerce_confidence(data["confidence"]),
            "reasoning": str(data["reasoning"]),
            "evidence": _coerce_str_list(data["evidence"]),
        }

    def _safe_default(self, state: PipelineState) -> Dict[str, Any]:
        # "keep" with 0.0 confidence: the controller's confident-revision branch is
        # disabled and the pragmatic-keep branch defers to the Polarity label.
        store = _seq_store(state)
        polarity = store.get("polarity", {})
        keep_label = polarity.get("label") or _neutral_or_first(state)
        return {
            "keep_or_revise": "keep",
            "final_label": keep_label,
            "confidence": 0.0,
            "reasoning": "coerced default (pragmatic parse failed)",
            "evidence": [],
            "coerced": True,
        }


# ---------------------------------------------------------------------------
# Stage 4 — Deterministic controller (no LLM)
# ---------------------------------------------------------------------------

def _neutral_or_first(state: PipelineState) -> str:
    labels = state.task_config.labels
    return "neutral" if "neutral" in labels else labels[0]


class SequentialController(BaseAgent[PipelineState]):
    """Deterministic Stage-4 controller: composes the three stage JSONs into a label.

    Precedence (first match wins), per the approved design:

    1. **No-opinion neutral** — intent says no opinion with confidence >= TAU_INTENT,
       UNLESS pragmatics confidently (>= TAU_REVISE) finds a non-neutral implicit
       stance (the escape hatch). This is the IntentGate, promoted to a first-class
       branch.
    2. **Confident pragmatic revision** — pragmatic revised with confidence >= TAU_REVISE
       -> pragmatic label.
    3. **Pragmatic keep** — pragmatic kept -> Polarity label.
    4. **Weak / conflicted** — a low-confidence revision is discarded; fall back to the
       primary when Polarity is also weak (< TAU_LOW) and USE_PRIMARY_FALLBACK is on,
       else to the Polarity label.

    Parameters
    ----------
    tau_intent, tau_revise, tau_low:
        Config-exposed thresholds (defaults 0.60 / 0.60 / 0.45).
    use_primary_fallback:
        When True (default) the primary participates as a safe fallback only
        (never a voter). Set False for a pure-sequential ablation.
    """

    def __init__(
        self,
        tau_intent: float = TAU_INTENT_DEFAULT,
        tau_revise: float = TAU_REVISE_DEFAULT,
        tau_low: float = TAU_LOW_DEFAULT,
        use_primary_fallback: bool = True,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name=name or "SequentialController", logger=logger)
        self.tau_intent = tau_intent
        self.tau_revise = tau_revise
        self.tau_low = tau_low
        self.use_primary_fallback = use_primary_fallback

    def validate_before(self, state: PipelineState) -> None:
        if not state.task_config.labels:
            raise ValueError("SequentialController: state.task_config.labels is empty.")

    def run(self, state: PipelineState) -> PipelineState:
        store = _seq_store(state)
        intent = store.get("intent", {}) or {}
        polarity = store.get("polarity", {}) or {}
        pragmatic = store.get("pragmatic", {}) or {}

        label, decided_by, fallback_path = self._decide(state, intent, polarity, pragmatic)

        # Coerce to a valid label as an absolute last resort.
        if not state.task_config.is_allowed_label(label):
            label = _neutral_or_first(state)
            fallback_path = "coerce_invalid_label"

        confidence = self._confidence_for(decided_by, polarity, pragmatic, state)

        store["thresholds"] = {
            "tau_intent": self.tau_intent,
            "tau_revise": self.tau_revise,
            "tau_low": self.tau_low,
            "use_primary_fallback": self.use_primary_fallback,
        }
        store["decided_by"] = decided_by
        store["fallback_path"] = fallback_path
        store["final_label"] = label

        rationale = (
            f"sequential_sentiment_v1: decided_by={decided_by}"
            + (f", fallback_path={fallback_path}" if fallback_path else "")
        )
        # Write both final_output and consensus_output so downstream consumers
        # (e.g. the explainability agent, evaluator) behave exactly as on the
        # parallel path.
        state.consensus_output = ConsensusOutput(
            label=label,
            confidence=confidence,
            votes={},
            rationale=rationale,
        )
        state.final_output = FinalOutput(
            label=label,
            confidence=confidence,
            payload={
                "source": "sequential_sentiment_v1",
                "decided_by": decided_by,
                "fallback_path": fallback_path,
            },
        )

        self.logger.debug(
            "%s: label=%s decided_by=%s fallback=%s", self.name, label, decided_by, fallback_path
        )
        state.append_history(
            component=self.name,
            summary=f"Sequential decision: '{label}' via {decided_by}.",
            outputs={
                "label": label,
                "confidence": confidence,
                "decided_by": decided_by,
                "fallback_path": fallback_path,
                "intent_opinion": intent.get("opinion_expressed"),
                "polarity_label": polarity.get("label"),
                "pragmatic_final_label": pragmatic.get("final_label"),
                "pragmatic_decision": pragmatic.get("keep_or_revise"),
            },
        )
        return state

    # -- decision rules ------------------------------------------------------

    def _decide(self, state, intent, polarity, pragmatic):
        """Return ``(label, decided_by, fallback_path)``."""
        neutral = _neutral_or_first(state)

        opinion = intent.get("opinion_expressed")
        intent_conf = _coerce_confidence(intent.get("confidence", 0.0))
        prag_decision = pragmatic.get("keep_or_revise")
        prag_label = pragmatic.get("final_label")
        prag_conf = _coerce_confidence(pragmatic.get("confidence", 0.0))
        pol_label = polarity.get("label")
        pol_conf = _coerce_confidence(polarity.get("confidence", 0.0))

        confident_implicit_stance = (
            prag_decision == "revise"
            and prag_label is not None
            and prag_label != neutral
            and prag_conf >= self.tau_revise
        )

        # Rule 1 — no-opinion neutral (with pragmatic escape hatch).
        if opinion is False and intent_conf >= self.tau_intent and not confident_implicit_stance:
            return neutral, "intent_no_opinion", None

        # Rule 2 — confident pragmatic revision.
        if prag_decision == "revise" and prag_conf >= self.tau_revise and prag_label is not None:
            return prag_label, "pragmatic_revision", None

        # Rule 3 — pragmatic keep -> Polarity label.
        if prag_decision == "keep":
            if pol_label is not None:
                return pol_label, "polarity_kept", None
            # Pragmatic kept but Polarity is unusable → treat as weak/conflicted.

        # Rule 4 — weak / conflicted fallback.
        return self._fallback(state, pol_label, pol_conf, neutral)

    def _fallback(self, state, pol_label, pol_conf, neutral):
        primary = state.primary_model_output
        primary_ok = (
            primary is not None
            and primary.label is not None
            and state.task_config.is_allowed_label(primary.label)
        )
        if self.use_primary_fallback and primary_ok and pol_conf < self.tau_low:
            return primary.label, "fallback_primary", "weak_polarity"
        if pol_label is not None:
            return pol_label, "fallback_polarity", "weak_revision"
        if primary_ok:
            return primary.label, "fallback_primary", "no_polarity"
        return neutral, "fallback_neutral", "no_polarity_no_primary"

    def _confidence_for(self, decided_by, polarity, pragmatic, state):
        if decided_by == "pragmatic_revision":
            return _coerce_confidence(pragmatic.get("confidence", 0.0))
        if decided_by in ("polarity_kept", "fallback_polarity"):
            return _coerce_confidence(polarity.get("confidence", 0.0))
        if decided_by == "fallback_primary":
            p = state.primary_model_output
            return p.confidence if p is not None and p.confidence is not None else 0.0
        if decided_by == "intent_no_opinion":
            # neutral decided by intent; report the pragmatic/keep confidence if any.
            return _coerce_confidence(pragmatic.get("confidence", 0.0))
        return 0.0
