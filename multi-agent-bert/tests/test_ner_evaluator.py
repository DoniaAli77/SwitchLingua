"""tests/test_ner_evaluator.py

Unit and integration tests for:
  - src/evaluation/ner_evaluator.py (NEREvaluator, NERReport, NERSampleResult,
    NERPerTagMetrics, _align_tags)
  - evaluate_pipeline.main() NER routing via --config (NER task guard removal)

Coverage areas
--------------
1. _align_tags helper — pad, truncate, equal length
2. NEREvaluator.evaluate() — token accuracy, per-tag metrics, macro F1
3. Prediction extraction — from sequence_output payload, missing key fallback
4. Pipeline exception handling — error captured, all-"O" fallback
5. Empty dataset raises ValueError
6. NEREvaluator.save() — correct file content for JSON/CSV predictions & metrics
7. end-to-end evaluate_pipeline.main() NER routing
8. Regression: existing Evaluator unchanged
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.ner_evaluator import (
    NEREvaluator,
    NERPerTagMetrics,
    NERReport,
    NERSampleResult,
    _align_tags,
)
from src.state.schema import FinalOutput, PipelineState, StateMetadata, TaskConfig

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def _ner_task_config(pipeline_mode: str = "paper_style") -> TaskConfig:
    return TaskConfig(
        task_name="ner",
        task_type="sequence_labeling",
        labels=_NER_LABELS,
        pipeline_mode=pipeline_mode,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Mock orchestrator
# ---------------------------------------------------------------------------


@dataclass
class _MockNEROrchestrator:
    """Injects a fixed sequence_output into state.final_output.

    Parameters
    ----------
    tag_fn : callable, optional
        Called as ``tag_fn(tokens) -> List[str]``.  Defaults to all-"O".
    raise_exc : bool
        When True, raises RuntimeError on every call (for error-path tests).
    omit_payload : bool
        When True, sets FinalOutput without a sequence_output key (stub mode).
    """

    tag_fn: Any = None
    raise_exc: bool = False
    omit_payload: bool = False

    def run(self, state: PipelineState) -> PipelineState:
        if self.raise_exc:
            raise RuntimeError("mock orchestrator failure")
        tokens: List[str] = state.extras.get("tokens") or state.input_text.split()
        if self.omit_payload:
            state.final_output = FinalOutput(
                payload={"source": "stub", "note": "primary_only stub"}
            )
            return state
        tags = self.tag_fn(tokens) if self.tag_fn else ["O"] * len(tokens)
        seq_out = [
            {"token": t, "tag": tag, "confidence": 1.0}
            for t, tag in zip(tokens, tags)
        ]
        state.final_output = FinalOutput(
            payload={"sequence_output": seq_out, "token_count": len(tags)}
        )
        return state


# ---------------------------------------------------------------------------
# Tiny dummy NER datasets
# ---------------------------------------------------------------------------

_SAMPLE_A = {
    "id": "s1",
    "text": "Ahmed works at Google",
    "tokens": ["Ahmed", "works", "at", "Google"],
    "tags": ["B-PER", "O", "O", "B-ORG"],
}
_SAMPLE_B = {
    "id": "s2",
    "text": "Sara joined Microsoft",
    "tokens": ["Sara", "joined", "Microsoft"],
    "tags": ["B-PER", "O", "B-ORG"],
}
_PERFECT_DATASET = [_SAMPLE_A, _SAMPLE_B]


def _perfect_tag_fn(tokens: List[str]) -> List[str]:
    """Return the gold tags for the two dummy samples based on token identity."""
    mapping = {
        "Ahmed": "B-PER",
        "Sara": "B-PER",
        "Google": "B-ORG",
        "Microsoft": "B-ORG",
    }
    return [mapping.get(t, "O") for t in tokens]


# ===========================================================================
# 1. _align_tags
# ===========================================================================


class TestAlignTags:
    def test_equal_length_unchanged(self):
        g, p = _align_tags(["B-PER", "O"], ["B-PER", "O"], 2)
        assert g == ["B-PER", "O"]
        assert p == ["B-PER", "O"]

    def test_gold_padded_with_O(self):
        g, p = _align_tags(["B-PER"], ["B-PER", "O"], 2)
        assert g == ["B-PER", "O"]
        assert len(p) == 2

    def test_pred_padded_with_O(self):
        g, p = _align_tags(["B-PER", "O"], ["B-PER"], 2)
        assert p == ["B-PER", "O"]

    def test_gold_truncated(self):
        g, _ = _align_tags(["B-PER", "O", "B-ORG"], ["O"], 2)
        assert len(g) == 2
        assert g == ["B-PER", "O"]

    def test_pred_truncated(self):
        _, p = _align_tags(["O"], ["B-PER", "O", "B-ORG"], 2)
        assert len(p) == 2

    def test_zero_length(self):
        g, p = _align_tags([], [], 0)
        assert g == []
        assert p == []

    def test_both_empty_padded_to_n(self):
        g, p = _align_tags([], [], 3)
        assert g == ["O", "O", "O"]
        assert p == ["O", "O", "O"]


# ===========================================================================
# 2. NEREvaluator.evaluate() — token accuracy, per-tag metrics, macro F1
# ===========================================================================


class TestNEREvaluatorEvaluate:
    def _run(self, dataset=None, tag_fn=None):
        dataset = dataset or _PERFECT_DATASET
        orch = _MockNEROrchestrator(tag_fn=tag_fn or _perfect_tag_fn)
        ev = NEREvaluator(task_config=_ner_task_config(), orchestrator=orch, run_id="t")
        return ev.evaluate(dataset)

    def test_returns_ner_report(self):
        report = self._run()
        assert isinstance(report, NERReport)

    def test_num_samples(self):
        report = self._run()
        assert report.num_samples == 2

    def test_num_tokens(self):
        # sample_A has 4, sample_B has 3 → 7
        report = self._run()
        assert report.num_tokens == 7

    def test_perfect_accuracy(self):
        report = self._run()
        assert report.token_accuracy == 1.0

    def test_zero_accuracy_all_wrong(self):
        def wrong_fn(tokens):
            return ["B-LOC"] * len(tokens)

        report = self._run(tag_fn=wrong_fn)
        assert report.token_accuracy == 0.0

    def test_partial_accuracy(self):
        # Predict correctly for 5 out of 7 tokens (Ahmed + Google + Sara correct,
        # but "at" predicted wrong, "works" predicted wrong)
        def partial_fn(tokens):
            m = {"Ahmed": "B-PER", "Google": "B-ORG", "Sara": "B-PER",
                 "Microsoft": "B-ORG"}
            return [m.get(t, "B-LOC") for t in tokens]  # "B-LOC" for O tokens → wrong

        report = self._run(tag_fn=partial_fn)
        # Ahmed→correct, works→wrong, at→wrong, Google→correct,
        # Sara→correct, joined→wrong, Microsoft→correct = 4/7
        assert report.token_accuracy == pytest.approx(4 / 7, rel=1e-4)

    def test_per_tag_contains_all_labels(self):
        report = self._run()
        tag_names = {m.label for m in report.per_tag}
        assert set(_NER_LABELS).issubset(tag_names)

    def test_per_tag_is_ner_per_tag_metrics(self):
        report = self._run()
        for m in report.per_tag:
            assert isinstance(m, NERPerTagMetrics)

    def test_macro_f1_is_mean_of_per_tag(self):
        report = self._run()
        expected = sum(m.f1 for m in report.per_tag) / len(report.per_tag)
        assert report.macro_f1 == pytest.approx(expected, rel=1e-5)

    def test_per_tag_b_per_perfect_f1(self):
        report = self._run()
        b_per = next(m for m in report.per_tag if m.label == "B-PER")
        assert b_per.f1 == pytest.approx(1.0)

    def test_per_tag_o_perfect_f1(self):
        report = self._run()
        o_tag = next(m for m in report.per_tag if m.label == "O")
        assert o_tag.f1 == pytest.approx(1.0)

    def test_per_tag_support_counts_gold(self):
        # B-PER: Ahmed + Sara = 2, B-ORG: Google + Microsoft = 2, O: works+at+joined = 3
        report = self._run()
        b_per = next(m for m in report.per_tag if m.label == "B-PER")
        assert b_per.support == 2
        b_org = next(m for m in report.per_tag if m.label == "B-ORG")
        assert b_org.support == 2
        o_tag = next(m for m in report.per_tag if m.label == "O")
        assert o_tag.support == 3

    def test_empty_dataset_raises(self):
        orch = _MockNEROrchestrator()
        ev = NEREvaluator(task_config=_ner_task_config(), orchestrator=orch)
        with pytest.raises(ValueError, match="empty"):
            ev.evaluate([])

    def test_samples_list_populated(self):
        report = self._run()
        assert len(report.samples) == 2

    def test_sample_result_fields(self):
        report = self._run()
        s = report.samples[0]
        assert isinstance(s, NERSampleResult)
        assert s.sample_id == "s1"
        assert s.tokens == ["Ahmed", "works", "at", "Google"]
        assert s.gold_tags == ["B-PER", "O", "O", "B-ORG"]
        assert s.predicted_tags == ["B-PER", "O", "O", "B-ORG"]
        assert s.token_count == 4
        assert s.correct_count == 4
        assert s.pipeline_error is None

    def test_run_id_in_report(self):
        report = self._run()
        assert report.run_id == "t"

    def test_timestamp_is_iso_string(self):
        report = self._run()
        # Should look like "2026-05-05T..."
        assert "T" in report.timestamp

    def test_meta_has_labels(self):
        report = self._run()
        assert "labels" in report.meta
        assert set(report.meta["labels"]) == set(_NER_LABELS)


# ===========================================================================
# 3. Prediction extraction — from payload and fallback
# ===========================================================================


class TestExtractPredictedTags:
    def _extract(self, state: PipelineState, n: int) -> List[str]:
        return NEREvaluator._extract_predicted_tags(state, n)

    def _make_state(self, tokens: List[str], tags: List[str]) -> PipelineState:
        seq_out = [
            {"token": t, "tag": tag, "confidence": 1.0}
            for t, tag in zip(tokens, tags)
        ]
        state = PipelineState(
            metadata=StateMetadata(sample_id="x"),
            input_text=" ".join(tokens),
            task_config=_ner_task_config(),
        )
        state.final_output = FinalOutput(
            payload={"sequence_output": seq_out, "token_count": len(tags)}
        )
        return state

    def test_extracts_tags_from_payload(self):
        state = self._make_state(["Ahmed", "works"], ["B-PER", "O"])
        tags = self._extract(state, 2)
        assert tags == ["B-PER", "O"]

    def test_missing_final_output_returns_all_O(self):
        state = PipelineState(
            metadata=StateMetadata(sample_id="x"),
            input_text="hello world",
            task_config=_ner_task_config(),
        )
        tags = self._extract(state, 2)
        assert tags == ["O", "O"]

    def test_final_output_without_sequence_key_returns_all_O(self):
        state = PipelineState(
            metadata=StateMetadata(sample_id="x"),
            input_text="hello",
            task_config=_ner_task_config(),
        )
        state.final_output = FinalOutput(
            payload={"source": "stub", "note": "primary_only stub"}
        )
        tags = self._extract(state, 3)
        assert tags == ["O", "O", "O"]

    def test_empty_sequence_output_list_returns_all_O(self):
        state = PipelineState(
            metadata=StateMetadata(sample_id="x"),
            input_text="hello",
            task_config=_ner_task_config(),
        )
        state.final_output = FinalOutput(payload={"sequence_output": []})
        tags = self._extract(state, 2)
        assert tags == ["O", "O"]

    def test_dict_entry_missing_tag_defaults_to_O(self):
        state = PipelineState(
            metadata=StateMetadata(sample_id="x"),
            input_text="hello",
            task_config=_ner_task_config(),
        )
        state.final_output = FinalOutput(
            payload={"sequence_output": [{"token": "hello", "confidence": 1.0}]}
        )
        tags = self._extract(state, 1)
        assert tags == ["O"]


# ===========================================================================
# 4. Pipeline exception handling
# ===========================================================================


class TestNEREvaluatorErrors:
    def test_pipeline_exception_sets_error_field(self):
        orch = _MockNEROrchestrator(raise_exc=True)
        ev = NEREvaluator(
            task_config=_ner_task_config(), orchestrator=orch, run_id="err"
        )
        report = ev.evaluate([_SAMPLE_A])
        assert report.samples[0].pipeline_error is not None

    def test_pipeline_exception_predicted_tags_all_O(self):
        orch = _MockNEROrchestrator(raise_exc=True)
        ev = NEREvaluator(task_config=_ner_task_config(), orchestrator=orch)
        report = ev.evaluate([_SAMPLE_A])
        assert all(t == "O" for t in report.samples[0].predicted_tags)

    def test_pipeline_exception_token_count_preserved(self):
        orch = _MockNEROrchestrator(raise_exc=True)
        ev = NEREvaluator(task_config=_ner_task_config(), orchestrator=orch)
        report = ev.evaluate([_SAMPLE_A])
        assert report.samples[0].token_count == len(_SAMPLE_A["tokens"])

    def test_stub_final_output_yields_all_O(self):
        """primary_only stub FinalOutput (no sequence_output) → all-O predictions."""
        orch = _MockNEROrchestrator(omit_payload=True)
        ev = NEREvaluator(task_config=_ner_task_config(), orchestrator=orch)
        report = ev.evaluate([_SAMPLE_A])
        assert all(t == "O" for t in report.samples[0].predicted_tags)
        assert report.samples[0].pipeline_error is None

    def test_error_samples_in_meta(self):
        orch = _MockNEROrchestrator(raise_exc=True)
        ev = NEREvaluator(task_config=_ner_task_config(), orchestrator=orch)
        report = ev.evaluate([_SAMPLE_A])
        assert report.meta["error_samples"] == 1

    def test_orchestrator_none_raises(self):
        with pytest.raises(ValueError, match="orchestrator"):
            NEREvaluator(task_config=_ner_task_config(), orchestrator=None)


# ===========================================================================
# 5. NEREvaluator.save() — file content
# ===========================================================================


class TestNEREvaluatorSave:
    def _make_report(self) -> NERReport:
        orch = _MockNEROrchestrator(tag_fn=_perfect_tag_fn)
        ev = NEREvaluator(
            task_config=_ner_task_config(), orchestrator=orch, run_id="test_run"
        )
        return ev.evaluate(_PERFECT_DATASET)

    def test_returns_four_path_keys(self):
        report = self._make_report()
        ev = NEREvaluator(
            task_config=_ner_task_config(),
            orchestrator=_MockNEROrchestrator(),
            run_id="test_run",
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = ev.save(report, output_dir=tmp)
        assert set(paths.keys()) == {
            "predictions_json",
            "predictions_csv",
            "metrics_json",
            "metrics_csv",
        }

    def test_predictions_json_keys(self):
        report = self._make_report()
        ev = NEREvaluator(
            task_config=_ner_task_config(),
            orchestrator=_MockNEROrchestrator(),
            run_id="test_run",
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = ev.save(report, output_dir=tmp)
            with open(paths["predictions_json"], encoding="utf-8") as fh:
                rows = json.load(fh)
        assert len(rows) == 2
        for row in rows:
            for key in ("id", "text", "tokens", "gold_tags", "predicted_tags"):
                assert key in row, f"missing key '{key}' in prediction row"

    def test_predictions_json_values(self):
        report = self._make_report()
        ev = NEREvaluator(
            task_config=_ner_task_config(),
            orchestrator=_MockNEROrchestrator(),
            run_id="test_run",
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = ev.save(report, output_dir=tmp)
            with open(paths["predictions_json"], encoding="utf-8") as fh:
                rows = json.load(fh)
        first = rows[0]
        assert first["id"] == "s1"
        assert first["tokens"] == ["Ahmed", "works", "at", "Google"]
        assert first["gold_tags"] == ["B-PER", "O", "O", "B-ORG"]
        assert first["predicted_tags"] == ["B-PER", "O", "O", "B-ORG"]

    def test_predictions_csv_columns(self):
        report = self._make_report()
        ev = NEREvaluator(
            task_config=_ner_task_config(),
            orchestrator=_MockNEROrchestrator(),
            run_id="test_run",
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = ev.save(report, output_dir=tmp)
            with open(paths["predictions_csv"], encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                headers = reader.fieldnames or []
                rows = list(reader)
        for col in ("id", "text", "tokens", "gold_tags", "predicted_tags"):
            assert col in headers
        assert len(rows) == 2

    def test_predictions_csv_tokens_json_encoded(self):
        """Tokens column should be JSON-encoded list, not raw Python repr."""
        report = self._make_report()
        ev = NEREvaluator(
            task_config=_ner_task_config(),
            orchestrator=_MockNEROrchestrator(),
            run_id="test_run",
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = ev.save(report, output_dir=tmp)
            with open(paths["predictions_csv"], encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
        tokens_cell = rows[0]["tokens"]
        parsed = json.loads(tokens_cell)
        assert parsed == ["Ahmed", "works", "at", "Google"]

    def test_metrics_json_top_level_keys(self):
        report = self._make_report()
        ev = NEREvaluator(
            task_config=_ner_task_config(),
            orchestrator=_MockNEROrchestrator(),
            run_id="test_run",
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = ev.save(report, output_dir=tmp)
            with open(paths["metrics_json"], encoding="utf-8") as fh:
                payload = json.load(fh)
        for key in ("token_accuracy", "macro_f1", "num_tokens", "per_tag", "meta"):
            assert key in payload

    def test_metrics_json_per_tag_entries(self):
        report = self._make_report()
        ev = NEREvaluator(
            task_config=_ner_task_config(),
            orchestrator=_MockNEROrchestrator(),
            run_id="test_run",
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = ev.save(report, output_dir=tmp)
            with open(paths["metrics_json"], encoding="utf-8") as fh:
                payload = json.load(fh)
        tags_in_file = {e["tag"] for e in payload["per_tag"]}
        assert "O" in tags_in_file
        assert "B-PER" in tags_in_file
        assert "B-ORG" in tags_in_file

    def test_metrics_csv_has_summary_row(self):
        report = self._make_report()
        ev = NEREvaluator(
            task_config=_ner_task_config(),
            orchestrator=_MockNEROrchestrator(),
            run_id="test_run",
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = ev.save(report, output_dir=tmp)
            with open(paths["metrics_csv"], encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
        summary = next((r for r in rows if r["tag"] == "__summary__"), None)
        assert summary is not None, "__summary__ row missing from metrics CSV"
        assert float(summary["token_accuracy"]) == pytest.approx(1.0)

    def test_metrics_csv_per_tag_rows(self):
        report = self._make_report()
        ev = NEREvaluator(
            task_config=_ner_task_config(),
            orchestrator=_MockNEROrchestrator(),
            run_id="test_run",
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = ev.save(report, output_dir=tmp)
            with open(paths["metrics_csv"], encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
        # One row per label + one summary row
        assert len(rows) == len(_NER_LABELS) + 1

    def test_output_dir_created_if_missing(self):
        report = self._make_report()
        ev = NEREvaluator(
            task_config=_ner_task_config(),
            orchestrator=_MockNEROrchestrator(),
            run_id="test_run",
        )
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = Path(tmp) / "deeply" / "nested" / "results"
            paths = ev.save(report, output_dir=str(new_dir))
            assert all(Path(p).exists() for p in paths.values())


# ===========================================================================
# 6. evaluate_pipeline.main() — NER routing via --config
# ===========================================================================


class TestEvaluatePipelineNERRouting:
    """Tests that main() correctly routes NER tasks to NEREvaluator."""

    def _write_ner_config(self, config_path: Path) -> None:
        """Write a minimal YAML config with ner task active."""
        config_content = """
active_task: ner
tasks:
  ner:
    task_type: sequence_labeling
    labels: [O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC]
    label_descriptions:
      O: Outside
      B-PER: Begin person
      I-PER: Inside person
      B-ORG: Begin organisation
      I-ORG: Inside organisation
      B-LOC: Begin location
      I-LOC: Inside location
execution:
  threshold: 0.6
  pipeline_mode: paper_style
  enable_deliberation: false
"""
        config_path.write_text(config_content, encoding="utf-8")

    def _write_ner_dataset(self, dataset_path: Path) -> None:
        """Write two NER samples in JSONL format."""
        samples = [
            {
                "id": "n1",
                "text": "Ahmed joined Google",
                "tokens": ["Ahmed", "joined", "Google"],
                "tags": ["B-PER", "O", "B-ORG"],
            },
            {
                "id": "n2",
                "text": "Sara works in Paris",
                "tokens": ["Sara", "works", "in", "Paris"],
                "tags": ["B-PER", "O", "O", "B-LOC"],
            },
        ]
        with open(dataset_path, "w", encoding="utf-8") as fh:
            for s in samples:
                fh.write(json.dumps(s) + "\n")

    def test_ner_routing_returns_zero(self):
        from evaluate_pipeline import main

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ner_config.yaml"
            dataset_path = Path(tmp) / "ner_data.jsonl"
            self._write_ner_config(config_path)
            self._write_ner_dataset(dataset_path)

            rc = main(
                [
                    "--dataset", str(dataset_path),
                    "--config", str(config_path),
                    "--output_dir", str(Path(tmp) / "results"),
                ]
            )
        assert rc == 0

    def test_ner_saves_predictions_json(self):
        from evaluate_pipeline import main

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ner_config.yaml"
            dataset_path = Path(tmp) / "ner_data.jsonl"
            out_dir = Path(tmp) / "results"
            self._write_ner_config(config_path)
            self._write_ner_dataset(dataset_path)

            main(
                [
                    "--dataset", str(dataset_path),
                    "--config", str(config_path),
                    "--output_dir", str(out_dir),
                ]
            )
            pred_files = list(out_dir.glob("*_ner_predictions.json"))
            assert len(pred_files) == 1
            rows = json.loads(pred_files[0].read_text(encoding="utf-8"))

        assert len(rows) == 2
        for row in rows:
            for key in ("id", "text", "tokens", "gold_tags", "predicted_tags"):
                assert key in row

    def test_ner_saves_metrics_json(self):
        from evaluate_pipeline import main

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ner_config.yaml"
            dataset_path = Path(tmp) / "ner_data.jsonl"
            out_dir = Path(tmp) / "results"
            self._write_ner_config(config_path)
            self._write_ner_dataset(dataset_path)

            main(
                [
                    "--dataset", str(dataset_path),
                    "--config", str(config_path),
                    "--output_dir", str(out_dir),
                ]
            )
            metrics_files = list(out_dir.glob("*_ner_metrics.json"))
            assert len(metrics_files) == 1
            payload = json.loads(metrics_files[0].read_text(encoding="utf-8"))

        assert "token_accuracy" in payload
        assert "macro_f1" in payload
        assert "per_tag" in payload

    def test_ner_ablation_rejected(self):
        """Ablation study must be rejected for NER tasks."""
        from evaluate_pipeline import main

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ner_config.yaml"
            dataset_path = Path(tmp) / "ner_data.jsonl"
            ablation_path = Path(tmp) / "ablation.yaml"
            self._write_ner_config(config_path)
            self._write_ner_dataset(dataset_path)
            ablation_path.write_text(
                "ablations:\n  - name: test\n", encoding="utf-8"
            )

            rc = main(
                [
                    "--dataset", str(dataset_path),
                    "--config", str(config_path),
                    "--output_dir", str(Path(tmp) / "results"),
                    "--ablation_config", str(ablation_path),
                ]
            )
        assert rc == 1

    def test_bad_ner_dataset_returns_one(self):
        """Non-existent dataset file should return exit code 1."""
        from evaluate_pipeline import main

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ner_config.yaml"
            self._write_ner_config(config_path)

            rc = main(
                [
                    "--dataset", str(Path(tmp) / "nonexistent.jsonl"),
                    "--config", str(config_path),
                    "--output_dir", str(Path(tmp) / "results"),
                ]
            )
        assert rc == 1


# ===========================================================================
# 7. Regression: existing Evaluator unchanged
# ===========================================================================


class TestExistingEvaluatorRegression:
    """Smoke-test to verify the classification Evaluator still works."""

    def _cls_dataset(self):
        return [
            {"id": "c1", "text": "great product", "label": "positive"},
            {"id": "c2", "text": "terrible", "label": "negative"},
        ]

    def test_classification_evaluator_runs(self):
        from src.agents.consensus_agent import ConsensusAgent
        from src.agents.contextual_agent import ContextualAgent
        from src.agents.explainability_agent import ExplainabilityAgent
        from src.agents.lexical_agent import LexicalAgent
        from src.agents.logic_agent import LogicAgent
        from src.evaluation.evaluator import Evaluator
        from src.llm.mock_client import MockLLMClient
        from src.models.mock_primary_classifier import MockPrimaryClassifier
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.pipeline.router import Router

        labels = ["positive", "negative", "neutral"]
        tc = TaskConfig(
            task_name="sentiment",
            labels=labels,
            threshold=0.99,
            pipeline_mode="paper_style",  # type: ignore[arg-type]
        )
        llm = MockLLMClient(mode="label_echo", allowed_labels=labels)
        orch = PipelineOrchestrator(
            primary_classifier=MockPrimaryClassifier(mode="heuristic"),
            router=Router(),
            lexical_agent=LexicalAgent(),
            contextual_agent=ContextualAgent(llm_client=llm),
            logic_agent=LogicAgent(),
            consensus_agent=ConsensusAgent(),
            explainability_agent=ExplainabilityAgent(),
        )
        ev = Evaluator(
            task_config=tc,
            orchestrator=orch,
            mode="full_pipeline",
            run_id="regression",
        )
        report = ev.evaluate(self._cls_dataset())
        assert report.accuracy >= 0.0
        assert report.macro_f1 >= 0.0
        assert len(report.per_class) == len(labels)
        for m in report.per_class:
            assert m.label in labels

    def test_existing_evaluator_save(self):
        from src.agents.consensus_agent import ConsensusAgent
        from src.agents.contextual_agent import ContextualAgent
        from src.agents.explainability_agent import ExplainabilityAgent
        from src.agents.lexical_agent import LexicalAgent
        from src.agents.logic_agent import LogicAgent
        from src.evaluation.evaluator import Evaluator
        from src.llm.mock_client import MockLLMClient
        from src.models.mock_primary_classifier import MockPrimaryClassifier
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.pipeline.router import Router

        labels = ["positive", "negative", "neutral"]
        tc = TaskConfig(
            task_name="sentiment",
            labels=labels,
            threshold=0.99,
            pipeline_mode="paper_style",  # type: ignore[arg-type]
        )
        llm = MockLLMClient(mode="label_echo", allowed_labels=labels)
        orch = PipelineOrchestrator(
            primary_classifier=MockPrimaryClassifier(mode="heuristic"),
            router=Router(),
            lexical_agent=LexicalAgent(),
            contextual_agent=ContextualAgent(llm_client=llm),
            logic_agent=LogicAgent(),
            consensus_agent=ConsensusAgent(),
            explainability_agent=ExplainabilityAgent(),
        )
        ev = Evaluator(
            task_config=tc,
            orchestrator=orch,
            mode="full_pipeline",
            run_id="regression_save",
        )
        report = ev.evaluate(self._cls_dataset())
        with tempfile.TemporaryDirectory() as tmp:
            paths = ev.save(report, output_dir=tmp)
        assert set(paths.keys()) == {
            "predictions_json", "predictions_csv", "metrics_json", "metrics_csv"
        }
        # No "ner_" prefix in paths
        for path in paths.values():
            assert "_ner_" not in path
