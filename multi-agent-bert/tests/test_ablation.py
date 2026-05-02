"""Unit tests for src/evaluation/ablation.py."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from typing import Dict, List, Optional

import pytest

from src.evaluation.ablation import (
    AblationConfig,
    AblationReport,
    AblationStudy,
    _DisabledAgent,
)
from src.state.schema import (
    FinalOutput,
    ModelOutput,
    PipelineState,
    RoutingInfo,
    StateMetadata,
    TaskConfig,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

LABELS = ["positive", "negative", "neutral"]


def make_task_config(labels=None) -> TaskConfig:
    lbs = labels or list(LABELS)
    return TaskConfig(
        task_name="test_task",
        labels=lbs,
        label_descriptions={lbl: lbl for lbl in lbs},
        threshold=0.65,
    )


def make_dataset(n: int = 6) -> List[Dict[str, str]]:
    texts_labels = [
        ("I love this", "positive"),
        ("terrible product", "negative"),
        ("it was okay", "neutral"),
        ("great experience", "positive"),
        ("awful", "negative"),
        ("fine I guess", "neutral"),
    ]
    return [
        {"id": f"s{i}", "text": t, "label": lbl}
        for i, (t, lbl) in enumerate(texts_labels[:n])
    ]


class _FakeOrchestrator:
    """Minimal orchestrator stub for ablation tests."""

    def __init__(self, label: str = "positive", escalated: bool = False):
        self._label = label
        self._escalated = escalated
        self._primary = _FakePrimary(label)

    def run(self, state: PipelineState) -> PipelineState:
        state.primary_model = ModelOutput(
            label=self._label,
            confidence=0.9,
            probabilities={lbl: (0.9 if lbl == self._label else 0.05)
                           for lbl in state.task_config.labels},
            raw_text=state.input_text,
        )
        decision = "escalate" if self._escalated else "accept_primary"
        state.routing_info = RoutingInfo(
            threshold=state.task_config.threshold,
            decision=decision,
        )
        state.final_output = FinalOutput(label=self._label, confidence=0.9)
        return state


class _FakePrimary:
    def __init__(self, label: str = "positive"):
        self._label = label

    def run(self, state: PipelineState) -> PipelineState:
        state.primary_model = ModelOutput(
            label=self._label,
            confidence=0.9,
            probabilities={lbl: (0.9 if lbl == self._label else 0.05)
                           for lbl in state.task_config.labels},
            raw_text=state.input_text,
        )
        return state


def make_study(label: str = "positive") -> AblationStudy:
    from src.llm.mock_client import MockLLMClient
    return AblationStudy(
        task_config=make_task_config(),
        primary_classifier=_FakePrimary(label),
        keyword_map={"positive": ["love", "great"], "negative": ["terrible", "awful"], "neutral": ["okay"]},
        rule_map={"positive": [r"\b(great|love)\b"], "negative": [r"\b(terrible|awful)\b"], "neutral": [r"\b(okay)\b"]},
        llm_client=MockLLMClient(mode="heuristic"),
        threshold=0.65,
        run_id="test_ablation",
    )


# ---------------------------------------------------------------------------
# AblationConfig — construction and helpers
# ---------------------------------------------------------------------------

class TestAblationConfig:
    def test_defaults(self):
        cfg = AblationConfig(name="base")
        assert cfg.use_lexical is True
        assert cfg.use_contextual is True
        assert cfg.use_logic is True
        assert cfg.use_deliberation is False
        assert cfg.consensus_weights is None
        assert cfg.threshold is None

    def test_from_dict_minimal(self):
        cfg = AblationConfig.from_dict({"name": "x"})
        assert cfg.name == "x"
        assert cfg.use_lexical is True

    def test_from_dict_full(self):
        cfg = AblationConfig.from_dict({
            "name": "custom",
            "use_lexical": False,
            "use_contextual": True,
            "use_logic": False,
            "use_deliberation": True,
            "consensus_weights": {"contextual": 2.0},
            "threshold": 0.7,
            "description": "test",
        })
        assert cfg.use_lexical is False
        assert cfg.use_logic is False
        assert cfg.use_deliberation is True
        assert cfg.consensus_weights == {"contextual": 2.0}
        assert cfg.threshold == 0.7

    def test_to_dict_round_trip(self):
        cfg = AblationConfig(
            name="roundtrip",
            use_lexical=False,
            description="rt",
        )
        restored = AblationConfig.from_dict(cfg.to_dict())
        assert restored.name == cfg.name
        assert restored.use_lexical == cfg.use_lexical
        assert restored.description == cfg.description

    def test_effective_weights_all_enabled(self):
        cfg = AblationConfig(name="all_on")
        w = cfg.effective_weights()
        # No explicit override — deliberation forced to 0.0 (disabled).
        assert w["deliberation"] == 0.0
        # Others not forced.
        assert "lexical" not in w or w["lexical"] != 0.0

    def test_effective_weights_disabled_lexical(self):
        cfg = AblationConfig(name="no_lex", use_lexical=False)
        w = cfg.effective_weights()
        assert w["lexical"] == 0.0

    def test_effective_weights_disabled_contextual(self):
        cfg = AblationConfig(name="no_ctx", use_contextual=False)
        w = cfg.effective_weights()
        assert w["contextual"] == 0.0

    def test_effective_weights_disabled_logic(self):
        cfg = AblationConfig(name="no_logic", use_logic=False)
        w = cfg.effective_weights()
        assert w["logic"] == 0.0

    def test_effective_weights_deliberation_enabled(self):
        cfg = AblationConfig(name="delib", use_deliberation=True)
        w = cfg.effective_weights()
        assert w["deliberation"] >= 1.0

    def test_effective_weights_explicit_override_respected(self):
        cfg = AblationConfig(
            name="custom_w",
            consensus_weights={"lexical": 3.0, "contextual": 0.5},
        )
        w = cfg.effective_weights()
        assert w["lexical"] == 3.0
        assert w["contextual"] == 0.5

    def test_effective_weights_disabled_overrides_explicit(self):
        """Disabled=True wins over explicit consensus weight."""
        cfg = AblationConfig(
            name="no_lex_explicit",
            use_lexical=False,
            consensus_weights={"lexical": 5.0},
        )
        w = cfg.effective_weights()
        assert w["lexical"] == 0.0


# ---------------------------------------------------------------------------
# AblationConfig — YAML / JSON loading
# ---------------------------------------------------------------------------

class TestAblationConfigLoading:
    def _write_yaml(self, tmpdir: str, content: str) -> str:
        path = os.path.join(tmpdir, "ablations.yaml")
        with open(path, "w") as f:
            f.write(content)
        return path

    def _write_json(self, tmpdir: str, data: dict) -> str:
        path = os.path.join(tmpdir, "ablations.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_load_yaml_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(tmpdir, """
ablations:
  - name: full_pipeline
    use_lexical: true
  - name: no_lexical
    use_lexical: false
""")
            configs = AblationConfig.load_yaml(path)
        assert len(configs) == 2
        assert configs[0].name == "full_pipeline"
        assert configs[1].use_lexical is False

    def test_load_yaml_with_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(tmpdir, """
ablations:
  - name: weighted
    consensus_weights:
      contextual: 2.0
      lexical: 0.5
""")
            configs = AblationConfig.load_yaml(path)
        assert configs[0].consensus_weights == {"contextual": 2.0, "lexical": 0.5}

    def test_load_yaml_duplicate_names_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(tmpdir, """
ablations:
  - name: dup
  - name: dup
""")
            with pytest.raises(ValueError, match="Duplicate"):
                AblationConfig.load_yaml(path)

    def test_load_yaml_missing_key_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(tmpdir, "just_a_string: true\n")
            with pytest.raises(ValueError, match="ablations"):
                AblationConfig.load_yaml(path)

    def test_load_json_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_json(tmpdir, {
                "ablations": [
                    {"name": "a", "use_lexical": True},
                    {"name": "b", "use_contextual": False},
                ]
            })
            configs = AblationConfig.load_json(path)
        assert len(configs) == 2
        assert configs[1].use_contextual is False

    def test_load_json_missing_key_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_json(tmpdir, {"other": []})
            with pytest.raises(ValueError, match="ablations"):
                AblationConfig.load_json(path)

    def test_load_yaml_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            AblationConfig.load_yaml("/nonexistent/path/ablations.yaml")


# ---------------------------------------------------------------------------
# _DisabledAgent
# ---------------------------------------------------------------------------

class TestDisabledAgent:
    def test_run_returns_state(self):
        agent = _DisabledAgent("lexical_agent")
        state = PipelineState(
            metadata=StateMetadata(sample_id="x"),
            input_text="hello",
            task_config=make_task_config(),
        )
        result = agent.run(state)
        assert result is state

    def test_run_appends_history(self):
        agent = _DisabledAgent("contextual_agent")
        state = PipelineState(
            metadata=StateMetadata(sample_id="x"),
            input_text="hello",
            task_config=make_task_config(),
        )
        agent.run(state)
        assert any(e.component == "contextual_agent" for e in state.history)

    def test_run_does_not_set_output_field(self):
        agent = _DisabledAgent("lexical_agent")
        state = PipelineState(
            metadata=StateMetadata(sample_id="x"),
            input_text="hello",
            task_config=make_task_config(),
        )
        agent.run(state)
        assert state.lexical_output is None


# ---------------------------------------------------------------------------
# AblationStudy — validation
# ---------------------------------------------------------------------------

class TestAblationStudyValidation:
    def test_empty_configs_raises(self):
        study = make_study()
        with pytest.raises(ValueError, match="non-empty"):
            study.run([], make_dataset())

    def test_empty_dataset_raises(self):
        study = make_study()
        cfg = AblationConfig(name="x")
        with pytest.raises(ValueError, match="empty"):
            study.run([cfg], [])

    def test_duplicate_config_names_raises(self):
        study = make_study()
        cfg1 = AblationConfig(name="dup")
        cfg2 = AblationConfig(name="dup")
        with pytest.raises(ValueError, match="Duplicate"):
            study.run([cfg1, cfg2], make_dataset())


# ---------------------------------------------------------------------------
# AblationStudy — results
# ---------------------------------------------------------------------------

class TestAblationStudyResults:
    def test_returns_ablation_report(self):
        study = make_study()
        configs = [AblationConfig(name="full")]
        report = study.run(configs, make_dataset())
        assert isinstance(report, AblationReport)

    def test_one_eval_report_per_config(self):
        study = make_study()
        configs = [
            AblationConfig(name="full"),
            AblationConfig(name="no_lexical", use_lexical=False),
        ]
        report = study.run(configs, make_dataset())
        assert len(report.reports) == 2

    def test_comparison_table_has_one_row_per_config(self):
        study = make_study()
        configs = [
            AblationConfig(name="a"),
            AblationConfig(name="b"),
            AblationConfig(name="c"),
        ]
        report = study.run(configs, make_dataset())
        assert len(report.comparison) == 3

    def test_comparison_row_keys(self):
        study = make_study()
        configs = [AblationConfig(name="full")]
        report = study.run(configs, make_dataset())
        row = report.comparison[0]
        assert "name" in row
        assert "accuracy" in row
        assert "macro_f1" in row
        assert "escalation_rate" in row
        assert "escalated_accuracy" in row
        # Per-class F1 columns.
        for lbl in LABELS:
            assert f"f1_{lbl}" in row

    def test_comparison_row_name_matches_config(self):
        study = make_study()
        configs = [
            AblationConfig(name="variant_A"),
            AblationConfig(name="variant_B", use_lexical=False),
        ]
        report = study.run(configs, make_dataset())
        names = [r["name"] for r in report.comparison]
        assert names == ["variant_A", "variant_B"]

    def test_configs_preserved_in_report(self):
        study = make_study()
        configs = [AblationConfig(name="x", use_contextual=False)]
        report = study.run(configs, make_dataset())
        assert report.configs[0].use_contextual is False

    def test_meta_contains_num_samples(self):
        study = make_study()
        dataset = make_dataset(4)
        report = study.run([AblationConfig(name="x")], dataset)
        assert report.meta["num_samples"] == 4

    def test_meta_contains_labels(self):
        study = make_study()
        report = study.run([AblationConfig(name="x")], make_dataset())
        assert report.meta["labels"] == LABELS

    def test_disabled_lexical_does_not_crash(self):
        study = make_study()
        cfg = AblationConfig(name="no_lex", use_lexical=False)
        report = study.run([cfg], make_dataset())
        assert report.reports[0].num_samples == 6

    def test_disabled_contextual_does_not_crash(self):
        study = make_study()
        cfg = AblationConfig(name="no_ctx", use_contextual=False)
        report = study.run([cfg], make_dataset())
        assert report.reports[0].num_samples == 6

    def test_disabled_logic_does_not_crash(self):
        study = make_study()
        cfg = AblationConfig(name="no_logic", use_logic=False)
        report = study.run([cfg], make_dataset())
        assert report.reports[0].num_samples == 6

    def test_threshold_override_applied(self):
        """Configs with threshold override use it; others use the base threshold."""
        study = make_study()
        cfg_low = AblationConfig(name="low_thresh", threshold=0.1)
        cfg_high = AblationConfig(name="high_thresh", threshold=0.99)
        report = study.run([cfg_low, cfg_high], make_dataset())
        # Both should complete without error.
        assert len(report.reports) == 2

    def test_consensus_weights_passed_to_agent(self):
        """Custom weights should not cause a crash."""
        study = make_study()
        cfg = AblationConfig(
            name="weighted",
            consensus_weights={"lexical": 2.0, "contextual": 0.5, "logic": 1.0},
        )
        report = study.run([cfg], make_dataset())
        assert report.reports[0].num_samples == 6


# ---------------------------------------------------------------------------
# AblationStudy — save()
# ---------------------------------------------------------------------------

class TestAblationStudySave:
    def _run_and_save(self, tmpdir: str, n_configs: int = 2):
        study = make_study()
        configs = [
            AblationConfig(name=f"cfg_{i}", use_lexical=(i % 2 == 0))
            for i in range(n_configs)
        ]
        report = study.run(configs, make_dataset())
        paths = study.save(report, output_dir=tmpdir)
        return report, paths

    def test_save_creates_comparison_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, paths = self._run_and_save(tmpdir)
            assert os.path.exists(paths["comparison_json"])

    def test_save_creates_comparison_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, paths = self._run_and_save(tmpdir)
            assert os.path.exists(paths["comparison_csv"])

    def test_comparison_json_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, paths = self._run_and_save(tmpdir, n_configs=2)
            with open(paths["comparison_json"], encoding="utf-8") as fh:
                data = json.load(fh)
            assert "comparison" in data
            assert "configs" in data
            assert len(data["comparison"]) == 2

    def test_comparison_csv_rows_match_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report, paths = self._run_and_save(tmpdir, n_configs=3)
            with open(paths["comparison_csv"], encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            assert len(rows) == 3

    def test_per_config_prediction_files_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, paths = self._run_and_save(tmpdir, n_configs=2)
            prediction_keys = [k for k in paths if k.endswith("__predictions_json")]
            assert len(prediction_keys) == 2

    def test_per_config_metrics_files_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, paths = self._run_and_save(tmpdir, n_configs=2)
            metrics_keys = [k for k in paths if k.endswith("__metrics_json")]
            assert len(metrics_keys) == 2

    def test_output_dir_created_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "nested", "output")
            study = make_study()
            report = study.run([AblationConfig(name="x")], make_dataset())
            study.save(report, output_dir=new_dir)
            assert os.path.isdir(new_dir)

    def test_run_id_override_in_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            study = make_study()
            report = study.run([AblationConfig(name="x")], make_dataset())
            paths = study.save(report, output_dir=tmpdir, run_id="custom_run")
            assert "custom_run" in paths["comparison_json"]

    def test_comparison_csv_has_f1_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, paths = self._run_and_save(tmpdir)
            with open(paths["comparison_csv"], encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            assert rows, "CSV is empty"
            for lbl in LABELS:
                assert f"f1_{lbl}" in rows[0], f"Missing f1_{lbl} column"
