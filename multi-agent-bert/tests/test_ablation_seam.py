"""Tests for the ablation seam: per-run control of w_primary via build_orchestrator.

Offline only. Confirms the default primary weight is unchanged (1.0) and that the
seam sets 0 / 1.5 etc. without touching consensus logic.
"""

from __future__ import annotations

import evaluate_pipeline
from src.state.schema import TaskConfig


def _tc() -> TaskConfig:
    return TaskConfig(
        task_name="sentiment",
        labels=["positive", "negative", "neutral"],
        label_descriptions={"positive": "p", "negative": "n", "neutral": "x"},
        threshold=0.6,
        pipeline_mode="full_agentic",
    )


def _orch(**kw):
    return evaluate_pipeline.build_orchestrator(
        task_config=_tc(), threshold=0.6, enable_deliberation=False, **kw
    )


def test_default_primary_weight_unchanged():
    # No seam → built-in default 1.0 (current behaviour preserved).
    assert _orch()._consensus.weights["primary"] == 1.0


def test_seam_sets_weight_zero():
    assert _orch(consensus_primary_weight=0.0)._consensus.weights["primary"] == 0.0


def test_seam_sets_weight_one_and_a_half():
    assert _orch(consensus_primary_weight=1.5)._consensus.weights["primary"] == 1.5


def test_seam_none_uses_default():
    assert _orch(consensus_primary_weight=None)._consensus.weights["primary"] == 1.0
