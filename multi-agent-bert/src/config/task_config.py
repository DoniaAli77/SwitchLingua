"""Dataclass-based configuration schema for a generic multi-agent text classification pipeline.

The pipeline is designed to operate over any language pair, including code-switched text
(e.g. Arabic-English, Hindi-English, Arabic-French).  Every agent can read the
``LanguagePairConfig`` nested inside ``PipelineConfig`` to adapt its behaviour:

* **LexicalAgent** – can select the correct monolingual or bilingual keyword list based on
  ``first_language_code`` / ``second_language_code`` and apply script-aware tokenization
  using ``first_script`` / ``second_script``.

* **ContextualAgent** – can adjust its system prompt to mention both languages, and handle
  intra-sentence vs. inter-sentence switching via ``code_switching_type``.

* **LogicAgent** – can apply language-specific consistency rules and flag cross-language
  contradictions only when ``code_switching_allowed`` is True.

* **ExplainabilityAgent** – can surface language-of-origin evidence per token when
  ``code_switching_allowed`` is True.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Language-pair configuration
# ---------------------------------------------------------------------------

CodeSwitchingType = Literal["intra_sentence", "inter_sentence", "intra_word", "none"]


@dataclass(slots=True)
class LanguagePairConfig:
    """Describes the language pair the pipeline should operate over.

    This is the single place agents inspect to adapt language-specific logic.

    Examples
    --------
    Arabic-English code-switched input::

        LanguagePairConfig(
            first_language="Arabic",
            second_language="English",
            first_language_code="ar",
            second_language_code="en",
            first_script="Arabic",
            second_script="Latin",
            code_switching_allowed=True,
            code_switching_type="intra_sentence",
        )

    Hindi-English inter-sentence switching::

        LanguagePairConfig("Hindi", "English", "hi", "en", "Devanagari", "Latin", True, "inter_sentence")

    Monolingual English (no switching)::

        LanguagePairConfig("English", "English", "en", "en", "Latin", "Latin", False, "none")
    """

    first_language: str
    second_language: str
    first_language_code: str
    second_language_code: str
    first_script: str
    second_script: str
    code_switching_allowed: bool = False
    code_switching_type: CodeSwitchingType = "none"

    @property
    def is_multilingual(self) -> bool:
        """True when the two language codes differ."""
        return self.first_language_code != self.second_language_code

    @property
    def language_pair_label(self) -> str:
        """Human-readable pair label, e.g. ``'Arabic-English'``."""
        return f"{self.first_language}-{self.second_language}"


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TaskConfig:
    """Core task settings shared across all pipeline components.

    Generic across topic classification, sentiment classification, and NER.
    Agents read ``labels`` to constrain their output and ``agent_weights``
    to know their relative contribution in the consensus step.
    """

    task_name: str
    labels: List[str] = field(default_factory=list)
    label_descriptions: Dict[str, str] = field(default_factory=dict)
    threshold: float = 0.5
    agent_weights: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent-specific sub-configurations
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LexicalConfig:
    """Lexical cues keyed by output label.

    Usage by LexicalAgent
    ---------------------
    The agent should look up keywords using the active label key.  When the
    pipeline is multilingual, keywords for each label can be extended at
    construction time with both L1 and L2 terms so the agent works regardless
    of which language a token belongs to.
    """

    keyword_map_by_label: Dict[str, List[str]] = field(default_factory=dict)


@dataclass(slots=True)
class LogicConfig:
    """Rule patterns keyed by output label.

    Usage by LogicAgent
    -------------------
    Each pattern string is a plain-English description of a consistency rule
    for that label.  When ``LanguagePairConfig.code_switching_allowed`` is
    True, the agent may also apply cross-language contradiction checks.
    """

    rule_patterns_by_label: Dict[str, List[str]] = field(default_factory=dict)


@dataclass(slots=True)
class PromptConfig:
    """Prompt templates for contextual and explainability agents.

    Usage by ContextualAgent
    ------------------------
    Inject ``{first_language}`` and ``{second_language}`` from
    ``LanguagePairConfig`` into the user prompt template so the LLM is aware
    of the active language pair and code-switching policy.

    Usage by ExplainabilityAgent
    ----------------------------
    Include ``{code_switching_type}`` in the explainability template when
    ``LanguagePairConfig.code_switching_allowed`` is True so explanations
    can surface per-language evidence.
    """

    contextual_system_prompt: str = "TODO: Define contextual agent system prompt."
    contextual_user_prompt_template: str = "TODO: Define contextual agent user prompt template."
    explainability_system_prompt: str = "TODO: Define explainability agent system prompt."
    explainability_user_prompt_template: str = "TODO: Define explainability agent user prompt template."


# ---------------------------------------------------------------------------
# Top-level pipeline configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PipelineConfig:
    """Full configuration bundle for one pipeline instantiation.

    ``PipelineConfig`` replaces ``MultiAgentTaskConfig`` and adds a mandatory
    ``language_pair`` field so every agent can access language metadata without
    passing extra arguments.

    Swapping language pair
    ----------------------
    Pass a different ``LanguagePairConfig`` to reuse the same task/lexical/logic
    settings for a different language pair::

        cfg = build_topic_classification_config(
            language_pair=LanguagePairConfig(
                "Arabic", "French", "ar", "fr", "Arabic", "Latin", True, "inter_sentence"
            )
        )
    """

    task: TaskConfig
    lexical: LexicalConfig
    logic: LogicConfig
    language_pair: LanguagePairConfig = field(
        default_factory=lambda: LanguagePairConfig(
            first_language="English",
            second_language="English",
            first_language_code="en",
            second_language_code="en",
            first_script="Latin",
            second_script="Latin",
            code_switching_allowed=False,
            code_switching_type="none",
        )
    )
    prompts: PromptConfig = field(default_factory=PromptConfig)


# Backward-compatible alias so existing imports of MultiAgentTaskConfig still work.
MultiAgentTaskConfig = PipelineConfig


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_topic_classification_config(
    language_pair: Optional[LanguagePairConfig] = None,
) -> PipelineConfig:
    """Return a fully populated topic-classification config.

    Parameters
    ----------
    language_pair:
        Override the default English-only pair.  Pass any ``LanguagePairConfig``
        to adapt lexical keywords and prompt templates to a different pair
        (e.g. Arabic-English, Hindi-English, Arabic-French).
    """

    labels = ["technology", "sports", "finance", "health", "politics"]

    task = TaskConfig(
        task_name="topic_classification",
        labels=labels,
        label_descriptions={
            "technology": "Articles about software, hardware, AI, and digital products.",
            "sports": "Content about teams, matches, athletes, and competitions.",
            "finance": "News on markets, banking, investing, and economic performance.",
            "health": "Topics about medicine, wellness, nutrition, and public health.",
            "politics": "Coverage of government, policy, elections, and diplomacy.",
        },
        threshold=0.65,
        agent_weights={
            "primary": 0.40,
            "lexical": 0.20,
            "contextual": 0.25,
            "logic": 0.15,
        },
    )

    # Default English keyword map.  When a multilingual pair is supplied,
    # L1/L2 keyword variants can be appended here in a real implementation.
    lexical = LexicalConfig(
        keyword_map_by_label={
            "technology": ["software", "startup", "AI", "cloud", "chip"],
            "sports": ["match", "league", "coach", "tournament", "goal"],
            "finance": ["stocks", "inflation", "earnings", "bank", "investor"],
            "health": ["vaccine", "hospital", "therapy", "diagnosis", "nutrition"],
            "politics": ["election", "policy", "parliament", "minister", "diplomacy"],
        }
    )

    logic = LogicConfig(
        rule_patterns_by_label={
            "technology": [
                "mentions product launches and engineering context",
                "contains platform or developer ecosystem references",
            ],
            "sports": [
                "describes competitive outcomes with teams or athletes",
                "includes event schedules, scores, or standings",
            ],
            "finance": [
                "discusses monetary metrics, markets, or investment decisions",
                "contains earnings reports, rates, or macroeconomic indicators",
            ],
            "health": [
                "focuses on medical interventions, symptoms, or care delivery",
                "references clinical findings or wellness guidance",
            ],
            "politics": [
                "centers on governance, policy decisions, or public institutions",
                "references political actors, legislation, or elections",
            ],
        }
    )

    active_pair = language_pair or LanguagePairConfig(
        first_language="English",
        second_language="English",
        first_language_code="en",
        second_language_code="en",
        first_script="Latin",
        second_script="Latin",
        code_switching_allowed=False,
        code_switching_type="none",
    )

    # Prompt templates reference language-pair placeholders so agents can
    # inject {first_language}/{second_language} at call time.
    prompts = PromptConfig(
        contextual_system_prompt=(
            "You are a contextual classifier for {first_language}"
            "{second_language_suffix}. "
            "Use discourse-level clues and intent to select the most appropriate task label."
        ),
        contextual_user_prompt_template=(
            "Task: {task_name}\n"
            "Language pair: {first_language} / {second_language}\n"
            "Code-switching: {code_switching_type}\n"
            "Labels: {labels}\n"
            "Text: {text}\n"
            "Return best label and short rationale."
        ),
        explainability_system_prompt=(
            "You are an explainability assistant for {first_language}"
            "{second_language_suffix} text. "
            "Summarize why the final label was chosen and mention conflicting signals, "
            "including language-of-origin evidence when code-switching is present."
        ),
        explainability_user_prompt_template=(
            "Task: {task_name}\n"
            "Language pair: {first_language} / {second_language}\n"
            "Code-switching type: {code_switching_type}\n"
            "Final label: {final_label}\n"
            "Agent evidence: {agent_evidence}\n"
            "Generate a concise, user-friendly explanation."
        ),
    )

    return PipelineConfig(
        task=task,
        lexical=lexical,
        logic=logic,
        language_pair=active_pair,
        prompts=prompts,
    )
