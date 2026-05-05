"""evaluate_pipeline.py — Command-line evaluation script for the multi-agent
text-classification pipeline.

Runs evaluation in one or both modes (primary-only, full-pipeline) over a
user-supplied JSONL dataset and writes metrics + predictions to an output
directory.  Also supports ablation studies driven by a YAML or JSON config.

Dataset format (one JSON object per line)
-----------------------------------------
{"id": "001", "text": "I love this product!", "label": "positive"}
{"id": "002", "text": "It was okay.",          "label": "neutral"}

The ``"id"`` field is optional; a sequential id will be auto-assigned when
absent.

Usage
-----
# Evaluate full pipeline with default mock components:
    python evaluate_pipeline.py --dataset data/eval.jsonl

# Evaluate primary model only:
    python evaluate_pipeline.py --dataset data/eval.jsonl --mode primary_only

# Evaluate full pipeline in paper-style mode (no contextual/deliberation):
    python evaluate_pipeline.py --dataset data/eval.jsonl --mode full_pipeline --pipeline_mode paper_style

# Both modes in one run:
    python evaluate_pipeline.py --dataset data/eval.jsonl --mode both

# Custom output directory and run id:
    python evaluate_pipeline.py \\
        --dataset data/eval.jsonl \\
        --output_dir results/ \\
        --run_id experiment_01

# Adjust routing threshold:
    python evaluate_pipeline.py --dataset data/eval.jsonl --threshold 0.7

# Ablation study (YAML config):
    python evaluate_pipeline.py --dataset data/eval.jsonl --ablation_config ablations.yaml

# Ablation study (JSON config):
    python evaluate_pipeline.py --dataset data/eval.jsonl --ablation_config ablations.json

Ablation YAML format
--------------------
    ablations:
      - name: full_pipeline
        description: All agents enabled
        use_lexical: true
        use_contextual: true
        use_logic: true
      - name: no_lexical
        description: Lexical agent disabled
        use_lexical: false
      - name: deliberation_on
        use_deliberation: true
        consensus_weights:
          deliberation: 1.5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict

# ---------------------------------------------------------------------------
# Path setup — allows running from the project root without installing.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.consensus_agent import ConsensusAgent
from src.agents.contextual_agent import ContextualAgent
from src.agents.deliberation_agent import DeliberationAgent
from src.agents.llm_explainability_agent import LLMExplainabilityAgent
from src.agents.llm_lexical_agent import LLMLexicalAgent
from src.agents.llm_logic_agent import LLMLogicAgent
from src.agents.ner_consensus_agent import NERConsensusAgent
from src.agents.ner_contextual_agent import NERContextualAgent
from src.agents.ner_lexical_agent import NERLexicalAgent
from src.agents.ner_logic_agent import NERLogicAgent
from src.agents.transformer_contextual_agent import TransformerContextualAgent
from src.agents.explainability_agent import ExplainabilityAgent
from src.agents.lexical_agent import LexicalAgent
from src.agents.logic_agent import LogicAgent
from src.config.loader import load_task_bundle
from src.evaluation.ablation import AblationConfig, AblationReport, AblationStudy
from src.evaluation.evaluator import EvalReport, Evaluator
from src.evaluation.ner_evaluator import NEREvaluator, NERReport
from src.llm.mock_client import MockLLMClient
from src.models.mock_primary_classifier import MockPrimaryClassifier
from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.router import Router
from src.state.schema import TaskConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate_pipeline")

# ---------------------------------------------------------------------------
# Default task configuration — override via CLI flags as needed.
# ---------------------------------------------------------------------------
_DEFAULT_LABELS = ["positive", "negative", "neutral"]
_DEFAULT_LABEL_DESCRIPTIONS = {
    # --- Sentiment labels ---
    "positive": "Text expressing positive or favorable sentiment.",
    "negative": "Text expressing negative or unfavorable sentiment.",
    "neutral":  "Text that is factual or carries no strong sentiment.",
    # --- Topic labels (Arabic-English bilingual, for code-switched classification) ---
    "business": (
        "Discussions about companies, startups, markets, products, mergers, and corporate strategy. "
        "شركة، سوق، منتج، استحواذ، اندماج، أرباح، CEO، عمل تجاري، مشروع."
    ),
    "education": (
        "Topics related to learning, schools, universities, courses, exams, assignments, and academic life. "
        "تعليم، جامعة، مدرسة، كورس، محاضرة، امتحان، واجب، طالب، ترم."
    ),
    "health": (
        "Conversations about physical wellness, fitness, exercise, diet, nutrition, vitamins, and healthy lifestyle. "
        "صحة، رياضة، تمارين، لياقة، تغذية، غذاء، فيتامين، نمط حياة صحي."
    ),
    "shopping": (
        "Buying goods online or in stores, orders, delivery, discounts, prices, and cart items. "
        "تسوق، شراء، متجر، محل، طلب، توصيل، خصم، تخفيضات، سعر، عربة."
    ),
    "medical": (
        "Clinical healthcare topics: doctors, patients, diagnoses, medications, symptoms, surgeries, and treatments. "
        "طبيب، دكتور، مريض، دواء، علاج، تشخيص، أشعة، عملية، موعد، مرض."
    ),
    "sports": (
        "Matches, teams, players, scores, goals, coaches, training sessions, and sporting events. "
        "رياضة، مباراة، فريق، لاعب، هدف، ماتش، مدرب، تدريب، نتيجة."
    ),
    "tech": (
        "Software, apps, AI, machine learning, cloud computing, updates, bugs, and technology systems. "
        "تكنولوجيا، برنامج، تطبيق، ذكاء اصطناعي، سحابة، تحديث، نظام، موديل."
    ),
    "finance": (
        "Banking, loans, investments, portfolios, currency, inflation, interest rates, and financial markets. "
        "بنك، قرض، فائدة، استثمار، محفظة، دولار، عملة، تضخم، ربح، مخاطرة."
    ),
    "social": (
        "Social media posts, platforms, communities, friend groups, meetups, and online interaction. "
        "تواصل اجتماعي، بوست، انستجرام، تويتر، واتساب، جروب، صديق، مجتمع، لقاء."
    ),
}
# Sentiment keyword / rule maps
_SENTIMENT_KEYWORD_MAP: Dict[str, List[str]] = {
    "positive": ["great", "excellent", "love", "amazing", "good"],
    "negative": ["terrible", "awful", "hate", "bad", "poor"],
    "neutral":  ["okay", "average", "fine", "normal"],
}
_SENTIMENT_RULE_MAP: Dict[str, List[str]] = {
    "positive": [r"\b(great|excellent|amazing|love)\b"],
    "negative": [r"\b(terrible|awful|hate|bad)\b"],
    "neutral":  [r"\b(okay|average|fine)\b"],
}

# Manually curated topic keyword knowledge for Arabic-English code-switched
# classification (Method 1: seed keyword maps).
#
# This operationalises the SwitchLingua paper's unspecified keyword-map
# construction step.  Each label has:
#   keywords_en  — English-script keywords for LexicalAgent
#   keywords_ar  — Arabic-script keywords for LexicalAgent
#   regex_rules  — Manually curated seed rules for LogicAgent (bilingual,
#                  2–4 per label).  Rules use re.IGNORECASE | re.UNICODE.
#                  An Arabic alternation rule is also auto-derived from
#                  keywords_ar and appended when building _TOPIC_RULE_MAP.
#
#   Seed rules follow the paper's description that LogicAgent applies
#   "task-specific pattern rules".  Since the paper does not specify how rules
#   are constructed, these are hand-crafted to:
#     (a) capture common single-topic keywords, and
#     (b) capture short bilingual patterns (Arabic word + English gloss or
#         vice-versa) typical of Arabic-English code-switching.
#   They can be refined or replaced once training/dev data is available.
#
# To replace keyword maps with data-driven equivalents run:
#   python scripts/build_keyword_map.py --input data/train.jsonl --output config/keyword_map.yaml
_TOPIC_KNOWLEDGE: Dict[str, Dict[str, List[str]]] = {
    "business": {
        "keywords_en": ["company", "startup", "market", "business", "customer",
                        "sales", "profit", "merger", "CEO", "product"],
        "keywords_ar": ["شركة", "شركات", "سوق", "عميل", "عملاء",
                        "مبيعات", "ربح", "أرباح", "إدارة", "منتج"],
        # Manually curated seed rules — can be refined from training/dev data.
        # Rules capture co-occurrence of bilingual term pairs typical of code-switching.
        "regex_rules": [
            r"(company|شركة|startup).*(market|سوق|product|منتج)",
            r"(profit|profits|ربح|أرباح).*(quarter|ربع|report|تقرير)",
            r"(CEO|مدير).*(merger|استحواذ|اندماج)",
        ],
    },
    "education": {
        "keywords_en": ["school", "university", "student", "students", "exam",
                        "course", "lecture", "assignment", "semester", "scholarship"],
        "keywords_ar": ["مدرسة", "جامعة", "طالب", "طلاب", "امتحان",
                        "اختبار", "محاضرة", "واجب", "ترم", "منحة"],
        # Manually curated seed rules — can be refined from training/dev data.
        "regex_rules": [
            r"(student|طلاب|طالب).*(course|كورس|lecture|محاضرة)",
            r"(exam|امتحان|اختبار).*(university|جامعة|school|مدرسة)",
            r"(assignment|واجب|semester|ترم).*(deadline|موعد|submit)",
        ],
    },
    "health": {
        "keywords_en": ["health", "fitness", "exercise", "diet", "vitamins",
                        "wellness", "sleep", "nutrition", "immune", "lifestyle"],
        "keywords_ar": ["صحة", "لياقة", "رياضة", "تمارين", "غذاء",
                        "فيتامين", "مناعة", "نوم", "تغذية", "نمط حياة"],
        # Manually curated seed rules — can be refined from training/dev data.
        "regex_rules": [
            r"(exercise|تمارين|رياضة).*(fitness|لياقة|health|صحة)",
            r"(diet|غذاء|nutrition|تغذية).*(healthy|صحي|lifestyle|نمط حياة)",
            r"(vitamin|فيتامين|immune|مناعة).*(health|صحة|body|جسم)",
        ],
    },
    "shopping": {
        "keywords_en": ["buy", "bought", "store", "shop", "shopping",
                        "order", "delivery", "discount", "cart", "price"],
        "keywords_ar": ["اشتريت", "شراء", "متجر", "محل", "طلب",
                        "توصيل", "خصم", "عربة", "سعر", "تخفيضات"],
        # Manually curated seed rules — can be refined from training/dev data.
        "regex_rules": [
            r"(buy|bought|اشتريت|شراء).*(store|shop|متجر|محل)",
            r"(order|طلب).*(delivery|توصيل|وصل)",
            r"(discount|خصم|تخفيضات).*(price|سعر|cart|عربة)",
        ],
    },
    "medical": {
        "keywords_en": ["doctor", "hospital", "patient", "medicine", "medication",
                        "treatment", "surgery", "scan", "diagnosis", "appointment"],
        "keywords_ar": ["طبيب", "دكتور", "مستشفى", "مريض", "دواء",
                        "علاج", "عملية", "أشعة", "تشخيص", "موعد"],
        # Manually curated seed rules — can be refined from training/dev data.
        "regex_rules": [
            r"(doctor|دكتور|طبيب).*(patient|مريض|appointment|موعد)",
            r"(medicine|medication|دواء|علاج).*(symptom|عرض|ألم|مرض)",
            r"(scan|أشعة|diagnosis|تشخيص).*(surgery|عملية|treatment|علاج)",
        ],
    },
    "sports": {
        "keywords_en": ["match", "team", "player", "goal", "coach",
                        "championship", "training", "football", "score", "gym"],
        "keywords_ar": ["ماتش", "مباراة", "فريق", "لاعب", "هدف",
                        "مدرب", "بطولة", "تدريب", "كورة", "ملعب"],
        # Manually curated seed rules — can be refined from training/dev data.
        "regex_rules": [
            r"(match|ماتش|مباراة).*(team|فريق|player|لاعب)",
            r"(goal|هدف|score).*(match|مباراة|game)",
            r"(coach|مدرب).*(training|تدريب|team|فريق)",
        ],
    },
    "tech": {
        "keywords_en": ["technology", "software", "app", "AI", "cloud",
                        "programming", "device", "smartphone", "update", "bugs"],
        "keywords_ar": ["تكنولوجيا", "تقنية", "برنامج", "تطبيق", "ذكاء اصطناعي",
                        "موبايل", "تحديث", "أجهزة", "برمجة", "سحابة"],
        # Manually curated seed rules — can be refined from training/dev data.
        "regex_rules": [
            r"(software|برنامج|app|تطبيق).*(update|تحديث|bug|bugs)",
            r"(AI|ذكاء اصطناعي|machine learning).*(system|نظام|model|موديل)",
            r"(cloud|سحابة).*(storage|تخزين|server|سيرفر)",
        ],
    },
    "finance": {
        "keywords_en": ["money", "bank", "loan", "investment", "portfolio",
                        "inflation", "dollar", "currency", "budget", "interest"],
        "keywords_ar": ["مال", "فلوس", "بنك", "قرض", "استثمار",
                        "محفظة", "تضخم", "دولار", "عملة", "ميزانية"],
        # Manually curated seed rules — can be refined from training/dev data.
        "regex_rules": [
            r"(bank|بنك).*(loan|قرض|interest|فائدة)",
            r"(investment|استثمار|portfolio|محفظة).*(profit|ربح|risk|مخاطرة)",
            r"(dollar|دولار|currency|عملة).*(inflation|تضخم|price|سعر)",
        ],
    },
    "social": {
        "keywords_en": ["social", "post", "Instagram", "Twitter", "WhatsApp",
                        "group", "friend", "community", "likes", "meetup"],
        "keywords_ar": ["اجتماعي", "بوست", "انستجرام", "تويتر", "واتساب",
                        "جروب", "صديق", "أصدقاء", "مجتمع", "لايكات"],
        # Manually curated seed rules — can be refined from training/dev data.
        "regex_rules": [
            r"(post|بوست).*(Instagram|انستجرام|Twitter|تويتر|likes|لايكات)",
            r"(WhatsApp|واتساب|group|جروب).*(friend|صديق|أصدقاء|community)",
            r"(meetup|لقاء|community|مجتمع).*(online|أونلاين|social|اجتماعي)",
        ],
    },
}

# Flatten _TOPIC_KNOWLEDGE into maps expected by build_agent_knowledge_maps().
# keywords_en + keywords_ar are merged into a single flat list per label.
# An Arabic alternation rule is auto-derived from keywords_ar and appended
# alongside the user-defined English regex_rules.
_TOPIC_KEYWORD_MAP: Dict[str, List[str]] = {
    label: d["keywords_en"] + d["keywords_ar"]
    for label, d in _TOPIC_KNOWLEDGE.items()
}
_TOPIC_RULE_MAP: Dict[str, List[str]] = {
    label: d["regex_rules"] + ["(" + "|".join(d["keywords_ar"]) + ")"]
    for label, d in _TOPIC_KNOWLEDGE.items()
}

# Backward-compat aliases (kept so any existing external references still work)
_DEFAULT_KEYWORD_MAP = _SENTIMENT_KEYWORD_MAP
_DEFAULT_RULE_MAP = _SENTIMENT_RULE_MAP


def build_agent_knowledge_maps(
    labels: List[str],
) -> tuple:
    """Return ``(keyword_map, rule_map)`` restricted to the active label list.

    Looks up each label in the merged pool of sentiment and topic maps.  Labels
    not found in any known map are silently omitted — agents handle an absent
    key gracefully.

    Parameters
    ----------
    labels:
        The active task label list from ``TaskConfig.labels``.

    Returns
    -------
    keyword_map:
        Mapping of label → keyword list for :class:`~src.agents.lexical_agent.LexicalAgent`.
    rule_map:
        Mapping of label → regex-pattern list for :class:`~src.agents.logic_agent.LogicAgent`.
    """
    all_keywords = {**_SENTIMENT_KEYWORD_MAP, **_TOPIC_KEYWORD_MAP}
    all_rules = {**_SENTIMENT_RULE_MAP, **_TOPIC_RULE_MAP}
    keyword_map = {lbl: all_keywords[lbl] for lbl in labels if lbl in all_keywords}
    rule_map = {lbl: all_rules[lbl] for lbl in labels if lbl in all_rules}
    return keyword_map, rule_map


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_classification_dataset(path: str) -> List[Dict[str, str]]:
    """Load a classification JSONL file (``text`` + ``label`` per line).

    Each line must be a valid JSON object with at minimum ``"text"`` and
    ``"label"`` keys.  The optional ``"id"`` key is used as the sample
    identifier.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file contains no valid samples.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    samples: List[Dict[str, str]] = []
    errors = 0
    with open(p, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Line %d — JSON parse error: %s", lineno, exc)
                errors += 1
                continue
            if "text" not in obj or "label" not in obj:
                log.warning(
                    "Line %d — missing 'text' or 'label' keys, skipping.", lineno
                )
                errors += 1
                continue
            samples.append(obj)

    if not samples:
        raise ValueError(
            f"No valid samples found in '{path}'. "
            f"{errors} line(s) had errors."
        )

    log.info("Loaded %d samples from '%s' (%d skipped).", len(samples), path, errors)
    return samples


def load_sequence_labeling_dataset(
    path: str,
    valid_tags: List[str],
) -> List[Dict]:
    """Load a sequence-labeling JSONL file (``text`` + ``tokens`` + ``tags``).

    Each line must be a valid JSON object with ``"text"``, ``"tokens"``, and
    ``"tags"`` keys.  The optional ``"id"`` key is used as the sample
    identifier.

    Validation per sample:

    * ``len(tokens) == len(tags)`` — must hold exactly.
    * Every tag must be present in ``valid_tags``.

    Lines that fail validation are skipped and counted as errors.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``valid_tags`` is empty, or the file contains no valid samples.
    """
    if not valid_tags:
        raise ValueError("valid_tags must not be empty.")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    valid_tag_set = set(valid_tags)
    samples: List[Dict] = []
    errors = 0
    with open(p, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Line %d — JSON parse error: %s", lineno, exc)
                errors += 1
                continue
            if "text" not in obj or "tokens" not in obj or "tags" not in obj:
                log.warning(
                    "Line %d — missing 'text', 'tokens', or 'tags' keys, skipping.",
                    lineno,
                )
                errors += 1
                continue
            tokens = obj["tokens"]
            tags = obj["tags"]
            if len(tokens) != len(tags):
                log.warning(
                    "Line %d — len(tokens)=%d != len(tags)=%d, skipping.",
                    lineno,
                    len(tokens),
                    len(tags),
                )
                errors += 1
                continue
            unknown = [t for t in tags if t not in valid_tag_set]
            if unknown:
                log.warning(
                    "Line %d — unknown tag(s) %s, skipping.",
                    lineno,
                    sorted(set(unknown)),
                )
                errors += 1
                continue
            samples.append(obj)

    if not samples:
        raise ValueError(
            f"No valid samples found in '{path}'. "
            f"{errors} line(s) had errors."
        )

    log.info("Loaded %d samples from '%s' (%d skipped).", len(samples), path, errors)
    return samples


def load_dataset(path: str) -> List[Dict[str, str]]:
    """Backward-compatible alias for :func:`load_classification_dataset`."""
    return load_classification_dataset(path)


# ---------------------------------------------------------------------------
# Orchestrator factory
# ---------------------------------------------------------------------------

def build_orchestrator(
    task_config: TaskConfig,
    threshold: float,
    enable_deliberation: bool,
    keyword_map: Dict[str, List[str]] | None = None,
    rule_map: Dict[str, List[str]] | None = None,
) -> PipelineOrchestrator:
    """Build a fully-wired orchestrator using mock components.

    Swap out ``MockPrimaryClassifier`` and ``MockLLMClient`` for real
    implementations when evaluating with actual model weights.

    ``ContextualAgent`` uses a ``label_echo`` client so it returns a valid
    task label from the prompt text.  ``DeliberationAgent`` (when enabled)
    uses a ``fixed`` client that returns a well-formed JSON response, because
    deliberation expects structured output that ``label_echo`` cannot produce.

    Parameters
    ----------
    keyword_map:
        Optional pre-built keyword map from the config loader.  When ``None``
        the legacy :func:`build_agent_knowledge_maps` is used as a fallback.
    rule_map:
        Optional pre-built rule map from the config loader.  When ``None``
        the legacy :func:`build_agent_knowledge_maps` is used as a fallback.
    """
    llm_client = MockLLMClient(
        mode="label_echo",
        allowed_labels=task_config.labels,
    )

    if enable_deliberation:
        import json as _json
        _delib_label = task_config.labels[0]
        _delib_response = _json.dumps({
            "recommended_label": _delib_label,
            "confidence": 0.75,
            "justification": "Mock deliberation: defaulting to first available label.",
            "mode": "recommendation",
        })
        deliberation_llm = MockLLMClient(
            mode="fixed",
            fixed_response=_delib_response,
        )
        deliberation_agent: DeliberationAgent | None = DeliberationAgent(llm_client=deliberation_llm)
    else:
        deliberation_agent = None

    # Use caller-supplied maps (from config loader) when available;
    # fall back to the legacy hardcoded knowledge maps otherwise.
    if keyword_map is None or rule_map is None:
        _kw, _rl = build_agent_knowledge_maps(task_config.labels)
        keyword_map = keyword_map if keyword_map is not None else _kw
        rule_map = rule_map if rule_map is not None else _rl

    paper_contextual_agent = TransformerContextualAgent(mode="tfidf")

    # LLM-backed specialist agents for full_agentic mode.
    # Both use the same label_echo client as the contextual agent so tests
    # can verify they are called without a real LLM backend.
    llm_lexical_agent = LLMLexicalAgent(llm_client=llm_client)
    llm_logic_agent = LLMLogicAgent(llm_client=llm_client)
    llm_explainability_agent = LLMExplainabilityAgent(llm_client=llm_client)

    # NER-path agents — wired with the task labels so they can validate tags.
    # Gazetteers / rule maps are intentionally empty here; override via
    # ner_lexical_agent / ner_logic_agent constructor kwargs when real entity
    # lists are available.
    ner_lexical     = NERLexicalAgent()
    ner_logic       = NERLogicAgent()
    ner_contextual  = NERContextualAgent()
    ner_consensus   = NERConsensusAgent()

    return PipelineOrchestrator(
        primary_classifier=MockPrimaryClassifier(mode="heuristic"),
        router=Router(),
        lexical_agent=LexicalAgent(keyword_map=keyword_map),
        contextual_agent=ContextualAgent(llm_client=llm_client),
        logic_agent=LogicAgent(rule_map=rule_map),
        consensus_agent=ConsensusAgent(),
        explainability_agent=ExplainabilityAgent(),
        deliberation_agent=deliberation_agent,
        paper_contextual_agent=paper_contextual_agent,
        llm_lexical_agent=llm_lexical_agent,
        llm_logic_agent=llm_logic_agent,
        llm_explainability_agent=llm_explainability_agent,
        ner_lexical_agent=ner_lexical,
        ner_logic_agent=ner_logic,
        ner_contextual_agent=ner_contextual,
        ner_consensus_agent=ner_consensus,
    )


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _safe_print(text: str) -> None:
    """Print text, replacing unencodable characters to avoid crashes on cp1252 terminals."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or "ascii", errors="replace").decode(sys.stdout.encoding or "ascii", errors="replace"))


def _print_ner_report(report: NERReport) -> None:
    sep = "─" * 60
    _safe_print(f"\n{sep}")
    _safe_print("  NER Evaluation Report")
    _safe_print(sep)
    _safe_print(f"  Run ID         : {report.run_id}")
    _safe_print(f"  Timestamp      : {report.timestamp}")
    _safe_print(f"  Samples        : {report.num_samples}")
    _safe_print(f"  Total tokens   : {report.num_tokens}")
    _safe_print(f"  Errors         : {report.meta.get('error_samples', 0)}")
    _safe_print(sep)
    _safe_print(f"  Token accuracy : {report.token_accuracy:.4f}")
    _safe_print(f"  Macro F1       : {report.macro_f1:.4f}")
    _safe_print(sep)
    _safe_print(f"  {'Tag':<12}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
    _safe_print(f"  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")
    for m in report.per_tag:
        _safe_print(
            f"  {m.label:<12}  {m.precision:>10.4f}  {m.recall:>8.4f}"
            f"  {m.f1:>8.4f}  {m.support:>8d}"
        )
    _safe_print(sep)


def _print_report(report: EvalReport) -> None:
    sep = "─" * 60
    _safe_print(f"\n{sep}")
    _safe_print(f"  Evaluation Report  [{report.mode.upper()}]")
    _safe_print(sep)
    _safe_print(f"  Run ID      : {report.run_id}")
    _safe_print(f"  Timestamp   : {report.timestamp}")
    _safe_print(f"  Samples     : {report.num_samples}")
    _safe_print(f"  Valid       : {report.meta.get('valid_samples', '?')}")
    _safe_print(f"  Errors      : {report.meta.get('error_samples', 0)}")
    _safe_print(sep)
    _safe_print(f"  Accuracy    : {report.accuracy:.4f}")
    _safe_print(f"  Macro F1    : {report.macro_f1:.4f}")
    _safe_print(sep)
    _safe_print(f"  Escalation rate     : {report.escalation_rate:.4f}")
    _safe_print(f"  Escalated count     : {report.escalated_count}")
    _safe_print(f"  Escalated accuracy  : {report.escalated_accuracy:.4f}")
    _safe_print(sep)
    _safe_print(f"  {'Label':<16}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
    _safe_print(f"  {'─'*16}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")
    for m in report.per_class:
        _safe_print(
            f"  {m.label:<16}  {m.precision:>10.4f}  {m.recall:>8.4f}"
            f"  {m.f1:>8.4f}  {m.support:>8d}"
        )
    _safe_print(sep)


def _print_ablation_report(report: AblationReport) -> None:
    sep = "─" * 80
    _safe_print(f"\n{sep}")
    _safe_print("  Ablation Study Comparison")
    _safe_print(sep)
    _safe_print(f"  Run ID    : {report.run_id}")
    _safe_print(f"  Timestamp : {report.timestamp}")
    _safe_print(f"  Configs   : {len(report.configs)}")
    _safe_print(f"  Samples   : {report.meta.get('num_samples', '?')}")
    _safe_print(sep)
    if not report.comparison:
        _safe_print("  (no results)")
        _safe_print(sep)
        return
    # Determine per-class F1 column names from first row.
    f1_cols = [k for k in report.comparison[0] if k.startswith("f1_")]
    col_w = 10
    header_parts = [
        f"  {'Name':<20}",
        f"{'Acc':>{col_w}}",
        f"{'MacroF1':>{col_w}}",
        f"{'EscRate':>{col_w}}",
        f"{'EscAcc':>{col_w}}",
    ] + [f"{c[3:]:>{col_w}}" for c in f1_cols]
    _safe_print("".join(header_parts))
    _safe_print(f"  {'─'*20}" + (f"  {'─'*(col_w-2)}" * (4 + len(f1_cols))))
    for row in report.comparison:
        row_parts = [
            f"  {row['name']:<20}",
            f"{row['accuracy']:>{col_w}.4f}",
            f"{row['macro_f1']:>{col_w}.4f}",
            f"{row['escalation_rate']:>{col_w}.4f}",
            f"{row['escalated_accuracy']:>{col_w}.4f}",
        ] + [f"{row.get(c, 0.0):>{col_w}.4f}" for c in f1_cols]
        _safe_print("".join(row_parts))
    _safe_print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the multi-agent text-classification pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        metavar="PATH",
        help="Path to a JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["primary_only", "full_pipeline", "both"],
        help=(
            "Evaluation mode. 'primary_only' scores the primary classifier alone. "
            "'full_pipeline' scores the final orchestrator output. "
            "'both' runs both modes. (default: both)"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        metavar="DIR",
        help="Directory to write JSON/CSV output files. (default: results/)",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        metavar="ID",
        help="Identifier for this run, used as the file-name prefix.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Router confidence threshold for escalation. Overrides config when set. (default: 0.65 or config value)",
    )
    parser.add_argument(
        "--pipeline_mode",
        default=None,
        choices=["primary_only", "paper_style", "full_agentic"],
        help=(
            "Pipeline execution mode used by the orchestrator. "
            "'primary_only' skips router and specialist agents, "
            "'paper_style' uses lexical+logic+contextual on escalation, no deliberation, "
            "'full_agentic' uses lexical+logic+contextual (and optional deliberation). "
            "Overrides config when set. (default: full_agentic or config value)"
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=(
            "Path to a pipeline YAML config file (e.g. src/config/default.yaml). "
            "When supplied, task labels, descriptions, and agent knowledge maps "
            "are read from the config.  CLI flags --threshold, --pipeline_mode, "
            "and --active_task override config values when explicitly provided."
        ),
    )
    parser.add_argument(
        "--active_task",
        default=None,
        metavar="TASK",
        help=(
            "Override the active_task key from the config file. "
            "Example: --active_task topic_classification"
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        help=(
            "Space-separated list of class labels used as a fallback when "
            "--config is not provided. Defaults to: positive negative neutral"
        ),
    )
    parser.add_argument(
        "--deliberation",
        action="store_true",
        default=False,
        help="Enable the optional deliberation stage on the escalation path.",
    )
    parser.add_argument(
        "--ablation_config",
        default=None,
        metavar="PATH",
        help=(
            "Path to a YAML or JSON file defining ablation variants. "
            "When supplied, a full ablation study is run and a comparison "
            "table is saved alongside per-config detail files. "
            "Mutually exclusive with --mode."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Set log level to DEBUG.",
    )

    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Build task config — must happen before loading the dataset so we
    # know whether to load a classification or NER dataset format.
    # ------------------------------------------------------------------
    keyword_map: Dict[str, List[str]] | None = None
    rule_map: Dict[str, List[str]] | None = None

    if args.config:
        try:
            bundle = load_task_bundle(
                args.config,
                active_task=args.active_task,
                # Pass CLI overrides only when they differ from argparse defaults,
                # i.e. when the user explicitly provided the flag.
                pipeline_mode=args.pipeline_mode if args.pipeline_mode is not None else None,
                threshold=args.threshold if args.threshold is not None else None,
                enable_deliberation=True if args.deliberation else None,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            log.error("Failed to load config '%s': %s", args.config, exc)
            return 1
        task_config = bundle.task_config
        keyword_map = bundle.keyword_map
        rule_map = bundle.rule_map
        log.info(
            "Config loaded — task=%s labels=%s pipeline_mode=%s threshold=%.2f",
            bundle.active_task,
            task_config.labels,
            task_config.pipeline_mode,
            task_config.threshold,
        )
    else:
        # Legacy path: build TaskConfig from CLI flags / hardcoded defaults.
        labels = args.labels if args.labels else list(_DEFAULT_LABELS)
        task_config = TaskConfig(
            task_name="evaluation",
            labels=labels,
            label_descriptions={
                lbl: _DEFAULT_LABEL_DESCRIPTIONS.get(lbl, lbl)
                for lbl in labels
            },
            threshold=args.threshold if args.threshold is not None else 0.65,
            enable_deliberation=args.deliberation,
            pipeline_mode=args.pipeline_mode if args.pipeline_mode is not None else "full_agentic",
        )

    # ------------------------------------------------------------------
    # NER path — sequence_labeling tasks get their own dataset loader
    # and evaluator.  Ablation is not supported for NER.
    # ------------------------------------------------------------------
    if task_config.task_type == "sequence_labeling":
        if args.ablation_config:
            log.error(
                "Ablation study is not supported for sequence_labeling tasks."
            )
            return 1
        try:
            ner_dataset = load_sequence_labeling_dataset(
                args.dataset, task_config.labels
            )
        except (FileNotFoundError, ValueError) as exc:
            log.error("%s", exc)
            return 1
        return _run_ner_evaluation(
            args=args,
            dataset=ner_dataset,
            task_config=task_config,
            keyword_map=keyword_map,
            rule_map=rule_map,
        )

    # ------------------------------------------------------------------
    # Classification path — load dataset now that we know the task type.
    # ------------------------------------------------------------------
    try:
        dataset = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    # ------------------------------------------------------------------
    # Ablation study path.
    # ------------------------------------------------------------------
    if args.ablation_config:
        return _run_ablation_study(
            args=args,
            dataset=dataset,
            task_config=task_config,
            keyword_map=keyword_map,
            rule_map=rule_map,
        )

    # ------------------------------------------------------------------
    # Standard evaluation (no ablation).
    # ------------------------------------------------------------------
    modes_to_run = (
        ["primary_only", "full_pipeline"]
        if args.mode == "both"
        else [args.mode]
    )

    orchestrator = build_orchestrator(
        task_config=task_config,
        threshold=task_config.threshold,
        enable_deliberation=task_config.enable_deliberation,
        keyword_map=keyword_map,
        rule_map=rule_map,
    )

    saved_paths: Dict[str, Dict[str, str]] = {}

    for mode in modes_to_run:
        run_id = f"{args.run_id}__{mode}" if args.run_id else None

        evaluator = Evaluator(
            task_config=task_config,
            orchestrator=orchestrator if mode == "full_pipeline" else None,
            primary_classifier=orchestrator._primary if mode == "primary_only" else None,
            mode=mode,
            run_id=run_id,
        )

        log.info("Running evaluation — mode=%s", mode)
        report = evaluator.evaluate(dataset)

        _print_report(report)

        paths = evaluator.save(report, output_dir=args.output_dir)
        saved_paths[mode] = paths

    _safe_print(f"\n{'─' * 60}")
    _safe_print("  Saved files:")
    for mode, paths in saved_paths.items():
        _safe_print(f"  [{mode}]")
        for key, path in paths.items():
            _safe_print(f"    {key}: {path}")
    _safe_print("─" * 60)

    return 0


def _run_ner_evaluation(
    args,
    dataset: List[Dict],
    task_config: TaskConfig,
    keyword_map: Dict[str, List[str]] | None = None,
    rule_map: Dict[str, List[str]] | None = None,
) -> int:
    """Run NER evaluation using :class:`~src.evaluation.ner_evaluator.NEREvaluator`."""
    orchestrator = build_orchestrator(
        task_config=task_config,
        threshold=task_config.threshold,
        enable_deliberation=task_config.enable_deliberation,
        keyword_map=keyword_map,
        rule_map=rule_map,
    )

    evaluator = NEREvaluator(
        task_config=task_config,
        orchestrator=orchestrator,
        run_id=args.run_id,
    )

    log.info("Running NER evaluation — %d samples", len(dataset))
    report = evaluator.evaluate(dataset)

    _print_ner_report(report)

    paths = evaluator.save(report, output_dir=args.output_dir)

    _safe_print(f"\n{'─' * 60}")
    _safe_print("  Saved files:")
    for key, path in paths.items():
        _safe_print(f"    {key}: {path}")
    _safe_print("─" * 60)

    return 0


def _run_ablation_study(
    args,
    dataset,
    task_config: TaskConfig,
    keyword_map: Dict[str, List[str]] | None = None,
    rule_map: Dict[str, List[str]] | None = None,
) -> int:
    """Load ablation configs and run a full ablation study."""
    path = args.ablation_config
    try:
        ext = Path(path).suffix.lower()
        if ext in (".yaml", ".yml"):
            configs = AblationConfig.load_yaml(path)
        elif ext == ".json":
            configs = AblationConfig.load_json(path)
        else:
            # Try YAML first, fall back to JSON.
            try:
                configs = AblationConfig.load_yaml(path)
            except Exception:
                configs = AblationConfig.load_json(path)
    except (FileNotFoundError, ValueError, Exception) as exc:
        log.error("Failed to load ablation config '%s': %s", path, exc)
        return 1

    log.info("Loaded %d ablation configs from '%s'.", len(configs), path)

    primary_clf = MockPrimaryClassifier(mode="heuristic")
    llm_client = MockLLMClient(mode="label_echo", allowed_labels=task_config.labels)
    if keyword_map is None or rule_map is None:
        _kw, _rl = build_agent_knowledge_maps(task_config.labels)
        keyword_map = keyword_map if keyword_map is not None else _kw
        rule_map = rule_map if rule_map is not None else _rl

    study = AblationStudy(
        task_config=task_config,
        primary_classifier=primary_clf,
        keyword_map=keyword_map,
        rule_map=rule_map,
        llm_client=llm_client,
        threshold=args.threshold,
        mode="full_pipeline",
        run_id=args.run_id,
    )

    log.info("Running ablation study — %d configs, %d samples.", len(configs), len(dataset))
    report = study.run(configs, dataset)

    _print_ablation_report(report)

    paths = study.save(report, output_dir=args.output_dir)

    _safe_print(f"\n{'─' * 60}")
    _safe_print("  Saved files:")
    for key, p in paths.items():
        _safe_print(f"    {key}: {p}")
    _safe_print("─" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
