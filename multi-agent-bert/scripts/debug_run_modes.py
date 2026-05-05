"""scripts/debug_run_modes.py

Runs the multi-agent classification pipeline on a JSONL dataset in all
three pipeline modes and prints the first 5 predictions per mode.

Usage
-----
    # Default (topic_classification, data/dev_dummy.jsonl):
    python scripts/debug_run_modes.py

    # Sentiment classification on the sentiment dev set:
    python scripts/debug_run_modes.py \\
        --config src/config/default.yaml \\
        --active_task sentiment_classification \\
        --dataset data/dev_dummy_sentiment.jsonl

    # Topic classification with a custom dataset:
    python scripts/debug_run_modes.py \\
        --active_task topic_classification \\
        --dataset data/dev_dummy.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow imports from the project root (multi-agent-bert/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate_pipeline import build_orchestrator
from src.config.loader import TaskBundle, load_task_bundle
from src.state.schema import PipelineState, StateMetadata

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_ROOT         = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG  = _ROOT / "src" / "config" / "default.yaml"
_DEFAULT_TASK    = "topic_classification"
_DEFAULT_DATASET = _ROOT / "data" / "dev_dummy.jsonl"
PREVIEW_COUNT = 5
PIPELINE_MODES = ["primary_only", "paper_style", "full_agentic"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug-run the multi-agent pipeline in all three modes.",
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help=f"Path to pipeline YAML config (default: {_DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--active_task",
        default=_DEFAULT_TASK,
        help=f"Active task name defined under 'tasks:' in the config (default: {_DEFAULT_TASK})",
    )
    parser.add_argument(
        "--dataset",
        default=str(_DEFAULT_DATASET),
        help=f"Path to a JSONL dataset (default: {_DEFAULT_DATASET})",
    )
    return parser.parse_args()





# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataset(path: Path) -> list[dict]:
    samples = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _run_single(orchestrator, task_config, sample: dict) -> tuple[dict, PipelineState | None]:
    """Run one sample through the orchestrator; return summary dict and state."""
    state = PipelineState(
        input_text=sample["text"],
        task_config=task_config,
        metadata=StateMetadata(sample_id=str(sample.get("id", "?"))),
    )
    try:
        state = orchestrator.run(state)
        final = state.final_output
        routing = state.routing_info
        predicted = final.label if final else "N/A"
        confidence = final.confidence if final else 0.0
        escalated = (routing.decision == "escalate") if routing else False
        error = None
    except Exception as exc:  # noqa: BLE001
        predicted = "ERROR"
        confidence = 0.0
        escalated = False
        error = str(exc)
        state = None

    summary = {
        "id":         sample.get("id", "?"),
        "true":       sample.get("label", "?"),
        "predicted":  predicted,
        "confidence": confidence,
        "escalated":  escalated,
        "error":      error,
    }
    return summary, state


_BAR  = "═" * 62
_DASH = "─" * 62


def _print_mode_banner(mode: str) -> None:
    print(f"\n{_BAR}")
    print(f"  PIPELINE MODE: {mode.upper()}")
    print(_BAR)


def _print_example_header(sample_id: str, true_label: str, index: int) -> None:
    print(f"\n  Example {index}  |  id={sample_id}  |  true label: {true_label}")
    print(f"  {_DASH}")


def _print_sample_summary(mode: str, state: PipelineState) -> None:
    final = state.final_output
    routing = state.routing_info
    label      = final.label if final else "N/A"
    confidence = f"{final.confidence:.3f}" if (final and final.confidence is not None) else "N/A"
    decision   = routing.decision if routing else "N/A"
    components = [e.component for e in state.history]
    print(f"  Mode       : {mode}")
    print(f"  Final label: {label}  (conf={confidence})")
    print(f"  Routing    : {decision}")
    print(f"  History    : {' → '.join(components) if components else '(empty)'}")
    print(f"  {_DASH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        sys.exit(1)

    samples = _load_dataset(dataset_path)
    preview = samples[:PREVIEW_COUNT]
    print(f"Config      : {args.config}")
    print(f"Active task : {args.active_task}")
    print(f"Dataset     : {dataset_path}  ({len(samples)} samples)")
    print(f"Showing first {PREVIEW_COUNT} per mode.\n")

    for mode in PIPELINE_MODES:
        # Load a fresh bundle for each mode so threshold/pipeline_mode override works.
        bundle: TaskBundle = load_task_bundle(
            args.config,
            active_task=args.active_task,
            pipeline_mode=mode,
            threshold=0.60,
        )
        task_config = bundle.task_config
        orch = build_orchestrator(
            task_config=task_config,
            threshold=0.60,
            enable_deliberation=False,
            keyword_map=bundle.keyword_map,
            rule_map=bundle.rule_map,
        )

        _print_mode_banner(mode)
        for idx, sample in enumerate(preview, start=1):
            result, state = _run_single(orch, task_config, sample)
            _print_example_header(
                sample_id=str(result["id"]),
                true_label=result["true"],
                index=idx,
            )
            if state is not None:
                _print_sample_summary(mode, state)
            else:
                print(f"  [ERROR] {result['error']}")
                print(f"  {_DASH}")

    print(f"\n{_BAR}")
    print("  Done.")
    print(f"{_BAR}\n")


if __name__ == "__main__":
    main()

