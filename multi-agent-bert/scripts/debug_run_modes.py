"""scripts/debug_run_modes.py

Runs the multi-agent classification pipeline on data/dev_dummy.jsonl in all
three pipeline modes and prints the first 5 predictions per mode.

Usage:
    python scripts/debug_run_modes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow imports from the project root (multi-agent-bert/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.consensus_agent import ConsensusAgent
from src.agents.contextual_agent import ContextualAgent
from src.agents.explainability_agent import ExplainabilityAgent
from src.agents.lexical_agent import LexicalAgent
from src.agents.logic_agent import LogicAgent
from src.llm.mock_client import MockLLMClient
from src.models.mock_primary_classifier import MockPrimaryClassifier
from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.router import Router
from src.state.schema import PipelineState, StateMetadata, TaskConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "dev_dummy.jsonl"
PREVIEW_COUNT = 5

LABELS = [
    "business", "education", "health", "shopping", "medical",
    "sports", "tech", "finance", "social",
]

KEYWORD_MAP: dict[str, list[str]] = {
    "business":  ["شركة", "company", "market", "business", "CEO", "investment"],
    "education": ["تعلم", "school", "curriculum", "scholarship", "students", "degree"],
    "health":    ["exercise", "fitness", "diet", "vitamins", "healthy"],
    "shopping":  ["اشتريت", "store", "delivery", "price", "تخفيضات"],
    "medical":   ["طبيب", "medication", "surgery", "scan", "patient"],
    "sports":    ["فريق", "championship", "player", "gym", "training"],
    "tech":      ["smartphone", "software", "cloud", "AI", "machine learning"],
    "finance":   ["dollar", "inflation", "portfolio", "loan", "bank"],
    "social":    ["post", "Instagram", "Twitter", "WhatsApp", "community"],
}

RULE_MAP: dict[str, list[str]] = {
    "business":  [r"\b(company|market|CEO|investment|merger|profit)\b"],
    "education": [r"\b(school|curriculum|scholarship|degree|students)\b"],
    "health":    [r"\b(exercise|fitness|diet|vitamins|healthy|lifestyle)\b"],
    "shopping":  [r"\b(store|delivery|price|discount|shopping)\b"],
    "medical":   [r"\b(medication|surgery|scan|patient|doctor|appointment)\b"],
    "sports":    [r"\b(championship|player|gym|training|match|season)\b"],
    "tech":      [r"\b(smartphone|software|cloud|AI|machine|learning|update)\b"],
    "finance":   [r"\b(dollar|inflation|portfolio|loan|bank|interest)\b"],
    "social":    [r"\b(post|Instagram|Twitter|WhatsApp|community|trending)\b"],
}

PIPELINE_MODES = ["primary_only", "paper_style", "full_agentic"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    samples = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def build_task_config(pipeline_mode: str) -> TaskConfig:
    return TaskConfig(
        task_name="debug_run",
        labels=LABELS,
        label_descriptions={lbl: lbl for lbl in LABELS},
        threshold=0.60,
        pipeline_mode=pipeline_mode,  # type: ignore[arg-type]
    )


def build_orchestrator(task_config: TaskConfig) -> PipelineOrchestrator:
    llm_client = MockLLMClient(mode="label_echo", allowed_labels=task_config.labels)
    return PipelineOrchestrator(
        primary_classifier=MockPrimaryClassifier(mode="heuristic"),
        router=Router(),
        lexical_agent=LexicalAgent(keyword_map=KEYWORD_MAP),
        contextual_agent=ContextualAgent(llm_client=llm_client),
        logic_agent=LogicAgent(rule_map=RULE_MAP),
        consensus_agent=ConsensusAgent(),
        explainability_agent=ExplainabilityAgent(),
        deliberation_agent=None,
    )


def run_single(
    orchestrator: PipelineOrchestrator,
    task_config: TaskConfig,
    sample: dict,
) -> tuple[dict, PipelineState | None]:
    """Run the orchestrator on one sample; return a summary dict and the state."""
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

    result = {
        "id":         sample.get("id", "?"),
        "true":       sample.get("label", "?"),
        "predicted":  predicted,
        "confidence": confidence,
        "escalated":  escalated,
        "error":      error,
    }
    return result, state


_BAR  = "═" * 62
_DASH = "─" * 62


def print_mode_banner(mode: str) -> None:
    print(f"\n{_BAR}")
    print(f"  PIPELINE MODE: {mode.upper()}")
    print(_BAR)


def print_example_header(sample_id: str, true_label: str, index: int) -> None:
    print(f"\n  Example {index}  |  id={sample_id}  |  true label: {true_label}")
    print(f"  {_DASH}")


def print_error(result: dict) -> None:
    print(f"  [ERROR] {result['error']}")
    print(f"  {_DASH}")


def _print_sample_summary(mode: str, state: PipelineState) -> None:
    """Print the four fields requested for each sample."""
    final = state.final_output
    routing = state.routing_info

    label      = final.label      if final   else "N/A"
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
    if not DATASET_PATH.exists():
        print(f"[ERROR] Dataset not found: {DATASET_PATH}")
        print("  Run:  python scripts/generate_dummy_data.py")
        sys.exit(1)

    samples = load_dataset(DATASET_PATH)
    preview = samples[:PREVIEW_COUNT]
    print(f"Loaded {len(samples)} samples — showing first {PREVIEW_COUNT} per mode.")

    for mode in PIPELINE_MODES:
        task_config = build_task_config(mode)
        orchestrator = build_orchestrator(task_config)

        print_mode_banner(mode)
        for idx, sample in enumerate(preview, start=1):
            result, state = run_single(orchestrator, task_config, sample)
            print_example_header(
                sample_id=str(result["id"]),
                true_label=result["true"],
                index=idx,
            )
            if state is not None:
                _print_sample_summary(mode, state)
            else:
                print_error(result)

    print(f"\n{_BAR}")
    print("  Done.")
    print(f"{_BAR}\n")


if __name__ == "__main__":
    main()
