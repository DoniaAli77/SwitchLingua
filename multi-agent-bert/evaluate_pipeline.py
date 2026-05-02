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
from src.agents.explainability_agent import ExplainabilityAgent
from src.agents.lexical_agent import LexicalAgent
from src.agents.logic_agent import LogicAgent
from src.evaluation.ablation import AblationConfig, AblationReport, AblationStudy
from src.evaluation.evaluator import EvalReport, Evaluator
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
    "positive": "Text expressing positive or favorable sentiment.",
    "negative": "Text expressing negative or unfavorable sentiment.",
    "neutral":  "Text that is factual or carries no strong sentiment.",
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

# Topic keyword / rule maps (Arabic-English code-switched)
_TOPIC_KEYWORD_MAP: Dict[str, List[str]] = {
    "business":  [
        "business", "company", "deal", "market", "profit", "revenue", "trade",
        "\u0634\u0631\u0643\u0629", "\u062a\u062c\u0627\u0631\u0629", "\u0633\u0648\u0642",
        "\u0631\u0628\u062d", "\u0635\u0641\u0642\u0629",
    ],
    "education": [
        "school", "university", "study", "learn", "teacher", "class", "course",
        "\u062a\u0639\u0644\u064a\u0645", "\u0645\u062f\u0631\u0633\u0629",
        "\u062c\u0627\u0645\u0639\u0629", "\u062f\u0631\u0627\u0633\u0629",
        "\u0637\u0627\u0644\u0628",
    ],
    "health":    [
        "health", "doctor", "exercise", "diet", "wellness", "hospital", "fit",
        "\u0635\u062d\u0629", "\u0645\u0633\u062a\u0634\u0641\u0649",
        "\u0637\u0628\u064a\u0628", "\u063a\u0630\u0627\u0621",
    ],
    "shopping":  [
        "shop", "buy", "purchase", "price", "discount", "sale", "product",
        "\u062a\u0633\u0648\u0642", "\u0634\u0631\u0627\u0621", "\u0633\u0639\u0631",
        "\u062e\u0635\u0645", "\u0645\u0646\u062a\u062c",
    ],
    "medical":   [
        "medicine", "treatment", "clinic", "prescription", "diagnosis", "symptoms", "drug",
        "\u0637\u0628", "\u0639\u0644\u0627\u062c", "\u0639\u064a\u0627\u062f\u0629",
        "\u062f\u0648\u0627\u0621", "\u0623\u0639\u0631\u0627\u0636",
    ],
    "sports":    [
        "sport", "football", "match", "team", "player", "score", "game",
        "\u0631\u064a\u0627\u0636\u0629", "\u0643\u0631\u0629", "\u0645\u0628\u0627\u0631\u0627\u0629",
        "\u0641\u0631\u064a\u0642", "\u0644\u0627\u0639\u0628",
    ],
    "tech":      [
        "tech", "software", "app", "code", "device", "digital", "computer", "AI",
        "\u062a\u0642\u0646\u064a\u0629", "\u062a\u0643\u0646\u0648\u0644\u0648\u062c\u064a\u0627",
        "\u0628\u0631\u0646\u0627\u0645\u062c", "\u062a\u0637\u0628\u064a\u0642",
        "\u0630\u0643\u0627\u0621",
    ],
    "finance":   [
        "finance", "money", "investment", "bank", "loan", "budget", "stock",
        "\u0645\u0627\u0644", "\u0627\u0633\u062a\u062b\u0645\u0627\u0631",
        "\u0628\u0646\u0643", "\u0642\u0631\u0636", "\u0645\u064a\u0632\u0627\u0646\u064a\u0629",
    ],
    "social":    [
        "social", "friend", "community", "media", "network", "post", "share",
        "\u0627\u062c\u062a\u0645\u0627\u0639\u064a", "\u0635\u062f\u064a\u0642",
        "\u0645\u062c\u062a\u0645\u0639", "\u062a\u0648\u0627\u0635\u0644",
    ],
}
_TOPIC_RULE_MAP: Dict[str, List[str]] = {
    "business":  [
        r"\b(business|company|trade|market|profit|revenue|deal)\b",
        r"(\u0634\u0631\u0643\u0629|\u062a\u062c\u0627\u0631\u0629|\u0633\u0648\u0642)",
    ],
    "education": [
        r"\b(school|university|study|learn|course|teacher)\b",
        r"(\u062a\u0639\u0644\u064a\u0645|\u0645\u062f\u0631\u0633\u0629|\u062c\u0627\u0645\u0639\u0629)",
    ],
    "health":    [
        r"\b(health|doctor|hospital|wellness|exercise|diet)\b",
        r"(\u0635\u062d\u0629|\u0645\u0633\u062a\u0634\u0641\u0649|\u0637\u0628\u064a\u0628)",
    ],
    "shopping":  [
        r"\b(shop|buy|purchase|price|discount|sale|product)\b",
        r"(\u062a\u0633\u0648\u0642|\u0634\u0631\u0627\u0621|\u062e\u0635\u0645)",
    ],
    "medical":   [
        r"\b(medicine|treatment|clinic|prescription|diagnosis|symptoms)\b",
        r"(\u0637\u0628|\u0639\u0644\u0627\u062c|\u0639\u064a\u0627\u062f\u0629)",
    ],
    "sports":    [
        r"\b(sport|football|match|team|player|score|game)\b",
        r"(\u0631\u064a\u0627\u0636\u0629|\u0643\u0631\u0629|\u0645\u0628\u0627\u0631\u0627\u0629)",
    ],
    "tech":      [
        r"\b(tech|software|app|code|device|digital|computer|AI)\b",
        r"(\u062a\u0642\u0646\u064a\u0629|\u062a\u0643\u0646\u0648\u0644\u0648\u062c\u064a\u0627|\u0628\u0631\u0646\u0627\u0645\u062c)",
    ],
    "finance":   [
        r"\b(finance|money|investment|bank|loan|budget|stock)\b",
        r"(\u0645\u0627\u0644|\u0627\u0633\u062a\u062b\u0645\u0627\u0631|\u0628\u0646\u0643)",
    ],
    "social":    [
        r"\b(social|friend|community|media|network|post|share)\b",
        r"(\u0627\u062c\u062a\u0645\u0627\u0639\u064a|\u0635\u062f\u064a\u0642|\u0645\u062c\u062a\u0645\u0639)",
    ],
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

def load_dataset(path: str) -> List[Dict[str, str]]:
    """Load a JSONL file and return a list of sample dicts.

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


# ---------------------------------------------------------------------------
# Orchestrator factory
# ---------------------------------------------------------------------------

def build_orchestrator(
    task_config: TaskConfig,
    threshold: float,
    enable_deliberation: bool,
) -> PipelineOrchestrator:
    """Build a fully-wired orchestrator using mock components.

    Swap out ``MockPrimaryClassifier`` and ``MockLLMClient`` for real
    implementations when evaluating with actual model weights.
    """
    llm_client = MockLLMClient(
        mode="label_echo",
        allowed_labels=task_config.labels,
    )
    deliberation_agent = DeliberationAgent() if enable_deliberation else None
    keyword_map, rule_map = build_agent_knowledge_maps(task_config.labels)

    return PipelineOrchestrator(
        primary_classifier=MockPrimaryClassifier(mode="heuristic"),
        router=Router(),
        lexical_agent=LexicalAgent(keyword_map=keyword_map),
        contextual_agent=ContextualAgent(llm_client=llm_client),
        logic_agent=LogicAgent(rule_map=rule_map),
        consensus_agent=ConsensusAgent(),
        explainability_agent=ExplainabilityAgent(),
        deliberation_agent=deliberation_agent,
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
        default=0.65,
        metavar="FLOAT",
        help="Router confidence threshold for escalation. (default: 0.65)",
    )
    parser.add_argument(
        "--pipeline_mode",
        default="full_agentic",
        choices=["primary_only", "paper_style", "full_agentic"],
        help=(
            "Pipeline execution mode used by the orchestrator. "
            "'primary_only' skips router and specialist agents, "
            "'paper_style' uses lexical+logic only on escalation, "
            "'full_agentic' uses lexical+logic+contextual (and optional deliberation). "
            "(default: full_agentic)"
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        help=(
            "Space-separated list of class labels. "
            "Defaults to: positive negative neutral"
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
    # Load dataset.
    # ------------------------------------------------------------------
    try:
        dataset = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    # ------------------------------------------------------------------
    # Build task config.
    # ------------------------------------------------------------------
    labels = args.labels if args.labels else list(_DEFAULT_LABELS)
    task_config = TaskConfig(
        task_name="evaluation",
        labels=labels,
        label_descriptions={
            lbl: _DEFAULT_LABEL_DESCRIPTIONS.get(lbl, lbl)
            for lbl in labels
        },
        threshold=args.threshold,
        enable_deliberation=args.deliberation,
        pipeline_mode=args.pipeline_mode,
    )

    # ------------------------------------------------------------------
    # Ablation study path.
    # ------------------------------------------------------------------
    if args.ablation_config:
        return _run_ablation_study(
            args=args,
            dataset=dataset,
            task_config=task_config,
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
        threshold=args.threshold,
        enable_deliberation=args.deliberation,
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


def _run_ablation_study(args, dataset, task_config: TaskConfig) -> int:
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
    keyword_map, rule_map = build_agent_knowledge_maps(task_config.labels)

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

    print(f"\n{'─' * 60}")
    print("  Saved files:")
    for key, p in paths.items():
        print(f"    {key}: {p}")
    print("─" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
