"""scripts/inspect_pipeline_modes.py

Print the concrete classes wired for each pipeline mode using
src/config/default.yaml, for active_task=topic_classification.

Usage:
    python scripts/inspect_pipeline_modes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate_pipeline import build_orchestrator, build_agent_knowledge_maps
from src.config.loader import load_task_bundle

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CONFIG = Path(__file__).parent.parent / "src" / "config" / "default.yaml"
_TASK = "topic_classification"

_SEP = "─" * 60


def _cls(obj) -> str:
    """Return 'ClassName' or '(none)' for None."""
    return type(obj).__name__ if obj is not None else "(none)"


def _print_mode(mode: str, o) -> None:
    """Print the wired classes for a single pipeline mode."""
    print(f"\n{_SEP}")
    print(f"  Mode : {mode}")
    print(_SEP)
    print(f"  primary_classifier          : {_cls(o._primary)}")

    if mode == "primary_only":
        print(f"  (router and specialist agents skipped in primary_only mode)")
        print(_SEP)
        return

    print(f"  router                      : {_cls(o._router)}")

    # Resolve which class *would be selected* for each role on escalation,
    # mirroring the selection logic in orchestrator.run() exactly.
    if mode == "paper_style":
        lexical_cls  = _cls(o._lexical)
        logic_cls    = _cls(o._logic)
        contextual_cls = _cls(
            o._paper_contextual if o._paper_contextual is not None else o._contextual
        )
        delib_cls    = "(not used in paper_style)"
        explain_cls  = _cls(o._explain)

    else:  # full_agentic
        lexical_cls  = _cls(
            o._llm_lexical if o._llm_lexical is not None else o._lexical
        )
        logic_cls    = _cls(
            o._llm_logic if o._llm_logic is not None else o._logic
        )
        contextual_cls = _cls(o._contextual)
        delib_cls    = (
            _cls(o._deliberation) if o._deliberation is not None else "(none — disabled)"
        )
        explain_cls  = _cls(
            o._llm_explain if o._llm_explain is not None else o._explain
        )

    print(f"  lexical_agent               : {lexical_cls}")
    print(f"  logic_agent                 : {logic_cls}")
    print(f"  contextual_agent            : {contextual_cls}")

    if mode == "paper_style":
        raw = o._paper_contextual
        print(f"  paper_contextual_agent      : {_cls(raw)}"
              + (" (fallback to contextual_agent)" if raw is None else ""))

    if mode == "full_agentic":
        print(f"  deliberation_agent          : {delib_cls}")

    print(f"  consensus_agent             : {_cls(o._consensus)}")
    print(f"  explainability_agent        : {explain_cls}")
    print(_SEP)


def main() -> None:
    if not _CONFIG.exists():
        print(f"ERROR: config not found at {_CONFIG}", file=sys.stderr)
        sys.exit(1)

    bundle = load_task_bundle(_CONFIG, active_task=_TASK)
    task_config = bundle.task_config
    keyword_map = bundle.keyword_map
    rule_map = bundle.rule_map

    print(f"\n{'═' * 60}")
    print(f"  Pipeline wiring inspection")
    print(f"  Config : {_CONFIG.relative_to(Path(__file__).parent.parent)}")
    print(f"  Task   : {_TASK}")
    print(f"  Labels : {task_config.labels}")
    print(f"{'═' * 60}")

    for mode in ("primary_only", "paper_style", "full_agentic"):
        task_config.pipeline_mode = mode  # type: ignore[assignment]
        enable_deliberation = (mode == "full_agentic")
        orchestrator = build_orchestrator(
            task_config=task_config,
            threshold=task_config.threshold,
            enable_deliberation=enable_deliberation,
            keyword_map=keyword_map,
            rule_map=rule_map,
        )
        _print_mode(mode, orchestrator)

    print()


if __name__ == "__main__":
    main()
