"""Unit tests for PrimaryTransformerClassifier.

All tests use a lightweight mock of the transformers + torch stack so they run
offline with no GPU and without installing HuggingFace dependencies.  The mock
is installed via ``monkeypatch`` / ``sys.modules`` patching before the module
under test is exercised.
"""

from __future__ import annotations

import sys
import types
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.state.schema import PipelineState, StateMetadata, TaskConfig


# ---------------------------------------------------------------------------
# Lightweight torch / transformers stubs
# ---------------------------------------------------------------------------

def _make_torch_stub():
    """Return a minimal torch-like module sufficient for the classifier."""
    import importlib

    torch_mod = types.ModuleType("torch")

    class _Tensor:
        def __init__(self, data: List[float]):
            self._data = list(data)

        def squeeze(self, dim: int) -> "_Tensor":
            return self

        def size(self, dim: int) -> int:
            return len(self._data)

        def argmax(self) -> "_Tensor":
            best = max(range(len(self._data)), key=lambda i: self._data[i])
            return _ScalarTensor(best)

        def to(self, device) -> "_Tensor":
            return self

        def __getitem__(self, idx):
            return _ScalarTensor(self._data[idx])

        def __setitem__(self, idx, val):
            self._data[idx] = val

        def item(self) -> float:
            return float(self._data[0])

    class _ScalarTensor(_Tensor):
        def __init__(self, val):
            self._val = val

        def item(self):
            return self._val

        def to(self, device):
            return self

    torch_mod.Tensor = _Tensor
    torch_mod.no_grad = MagicMock(return_value=_NoGradCtx())

    nn_mod = types.ModuleType("torch.nn")
    functional_mod = types.ModuleType("torch.nn.functional")

    def _softmax(tensor, dim):
        import math
        data = tensor._data
        # Handle -inf masking.
        max_val = max(v for v in data if v != float("-inf"))
        exps = [math.exp(v - max_val) if v != float("-inf") else 0.0 for v in data]
        total = sum(exps)
        result = _Tensor([e / total for e in exps])
        return result

    functional_mod.softmax = _softmax
    nn_mod.functional = functional_mod
    torch_mod.nn = nn_mod

    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = functional_mod
    return torch_mod


class _NoGradCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _make_transformers_stub(logits: List[float], id2label: Dict[int, str]):
    """Return minimal transformers stubs with controllable logit output."""
    transformers_mod = types.ModuleType("transformers")

    class _FakeOutput:
        def __init__(self):
            import sys
            torch_mod = sys.modules["torch"]
            self.logits = torch_mod.Tensor(logits)
            # Shape: [1, num_labels] — squeeze() is called in predict()
            self.logits = _BatchedTensor(logits)

    class _BatchedTensor:
        """Simulates logits of shape [1, num_labels]."""
        def __init__(self, data):
            import sys
            torch_mod = sys.modules["torch"]
            self._inner = torch_mod.Tensor(data)

        def squeeze(self, dim):
            return self._inner

        def to(self, device):
            return self

    class _FakeConfig:
        def __init__(self):
            self.id2label = {str(k): v for k, v in id2label.items()}

    class _FakeModel:
        def __init__(self):
            self.config = _FakeConfig()
            self._device = "cpu"

        def to(self, device):
            self._device = device
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return _FakeOutput()

    class _InputValue:
        """Wrapper so tokenizer outputs have a .to() method."""
        def __init__(self, val):
            self._val = val

        def to(self, device):
            return self

    class _FakeTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": _InputValue([[1, 2, 3]])}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(checkpoint):
            return _FakeTokenizer()

    class FakeAutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(checkpoint, **kwargs):
            return _FakeModel()

    transformers_mod.AutoTokenizer = FakeAutoTokenizer
    transformers_mod.AutoModelForSequenceClassification = FakeAutoModelForSequenceClassification

    sys.modules["transformers"] = transformers_mod
    return transformers_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LABELS = ["positive", "negative", "neutral"]


@pytest.fixture(autouse=True)
def _patch_torch_and_transformers(monkeypatch):
    """Install torch + transformers stubs before each test; restore after."""
    original = {k: sys.modules.get(k) for k in ("torch", "torch.nn", "torch.nn.functional", "transformers")}
    _make_torch_stub()
    yield
    for k, v in original.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def _make_classifier(logits=None, id2label=None, label_map=None):
    """Return a loaded PrimaryTransformerClassifier backed by stubs."""
    if logits is None:
        logits = [2.0, 0.5, 0.1]  # positive wins
    if id2label is None:
        id2label = {0: "positive", 1: "negative", 2: "neutral"}
    _make_transformers_stub(logits, id2label)

    from src.models.primary_transformer_classifier import PrimaryTransformerClassifier

    clf = PrimaryTransformerClassifier(
        checkpoint="fake-model",
        label_map=label_map,
        device="cpu",
    )
    clf.load()
    return clf


def make_state(text: str = "I love this product", labels=None) -> PipelineState:
    return PipelineState(
        metadata=StateMetadata(sample_id="pt-001"),
        input_text=text,
        task_config=TaskConfig(
            task_name="sentiment",
            labels=labels if labels is not None else list(LABELS),
            label_descriptions={
                "positive": "Positive sentiment.",
                "negative": "Negative sentiment.",
                "neutral": "Neutral sentiment.",
            },
        ),
    )


# ---------------------------------------------------------------------------
# load() tests
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_sets_loaded_flag(self):
        clf = _make_classifier()
        assert clf._loaded is True

    def test_load_is_idempotent(self):
        clf = _make_classifier()
        clf.load()   # second call should be a no-op
        assert clf._loaded is True

    def test_load_raises_import_error_if_transformers_missing(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "transformers", raising=False)
        # Patch the import inside the module to raise.
        from src.models import primary_transformer_classifier as ptc_mod
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else None

        import builtins
        original = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("mocked missing transformers")
            return original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        from src.models.primary_transformer_classifier import PrimaryTransformerClassifier
        clf = PrimaryTransformerClassifier(checkpoint="fake", label_map={0: "positive"})
        clf._loaded = False  # force reload path
        with pytest.raises(ImportError, match="transformers"):
            clf.load()

    def test_from_pretrained_returns_loaded_instance(self):
        _make_transformers_stub([1.0, 0.2, 0.1], {0: "positive", 1: "negative", 2: "neutral"})
        from src.models.primary_transformer_classifier import PrimaryTransformerClassifier
        clf = PrimaryTransformerClassifier.from_pretrained(
            checkpoint="fake-model",
            label_map={0: "positive", 1: "negative", 2: "neutral"},
        )
        assert clf._loaded is True

    def test_label_map_built_from_id2label_when_none_given(self):
        clf = _make_classifier(label_map=None)
        assert clf.label_map is not None
        assert clf.label_map[0] == "positive"

    def test_explicit_label_map_overrides_id2label(self):
        clf = _make_classifier(label_map={0: "CUSTOM_NEG", 1: "CUSTOM_POS", 2: "CUSTOM_NEU"})
        # label_map supplied at construction — id2label should not overwrite it.
        assert clf.label_map[0] == "CUSTOM_NEG"


# ---------------------------------------------------------------------------
# predict() tests
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_returns_model_output(self):
        clf = _make_classifier(logits=[2.0, 0.5, 0.1])
        output = clf.predict("some text", task_labels=LABELS)
        assert output.label is not None
        assert output.confidence is not None
        assert output.probabilities != {}

    def test_predict_winning_label_matches_highest_logit(self):
        # positive has highest logit
        clf = _make_classifier(logits=[2.0, 0.5, 0.1])
        output = clf.predict("I love it", task_labels=LABELS)
        assert output.label == "positive"

    def test_predict_second_label_wins_when_logit_is_highest(self):
        # negative has highest logit
        clf = _make_classifier(logits=[0.1, 3.0, 0.5])
        output = clf.predict("I hate it", task_labels=LABELS)
        assert output.label == "negative"

    def test_probabilities_sum_to_one(self):
        clf = _make_classifier(logits=[1.0, 0.5, 0.2])
        output = clf.predict("test", task_labels=LABELS)
        assert abs(sum(output.probabilities.values()) - 1.0) < 1e-4

    def test_probabilities_keys_match_task_labels(self):
        clf = _make_classifier(logits=[1.0, 0.5, 0.2])
        output = clf.predict("test", task_labels=LABELS)
        assert set(output.probabilities.keys()) == set(LABELS)

    def test_confidence_matches_winning_label_probability(self):
        clf = _make_classifier(logits=[2.0, 0.5, 0.1])
        output = clf.predict("test", task_labels=LABELS)
        assert output.confidence == pytest.approx(output.probabilities[output.label], abs=1e-5)

    def test_raw_text_stored(self):
        clf = _make_classifier(logits=[2.0, 0.5, 0.1])
        output = clf.predict("Hello world", task_labels=LABELS)
        assert output.raw_text == "Hello world"

    def test_predict_raises_if_not_loaded(self):
        _make_transformers_stub([1.0, 0.0, 0.0], {0: "positive", 1: "negative", 2: "neutral"})
        from src.models.primary_transformer_classifier import PrimaryTransformerClassifier
        clf = PrimaryTransformerClassifier(checkpoint="fake", label_map={0: "positive"})
        with pytest.raises(RuntimeError, match="not loaded"):
            clf.predict("text")

    def test_task_label_masking_excludes_unlisted_labels(self):
        """When task_labels is a subset of model outputs, excluded labels get prob ≈ 0."""
        clf = _make_classifier(logits=[2.0, 3.0, 0.1])
        # Only allow positive and neutral — negative should be masked out.
        output = clf.predict("test", task_labels=["positive", "neutral"])
        assert "negative" not in output.probabilities
        assert set(output.probabilities.keys()) == {"positive", "neutral"}

    def test_predict_no_task_labels_uses_all_classes(self):
        clf = _make_classifier(logits=[0.1, 3.0, 0.5])
        output = clf.predict("test", task_labels=None)
        # All three labels should be present.
        assert len(output.probabilities) == 3


# ---------------------------------------------------------------------------
# run() tests (orchestrator interface)
# ---------------------------------------------------------------------------

class TestRun:
    def test_run_writes_primary_model_output(self):
        clf = _make_classifier(logits=[2.0, 0.5, 0.1])
        state = make_state()
        state = clf.run(state)
        assert state.primary_model_output.label is not None

    def test_run_label_matches_predict(self):
        clf = _make_classifier(logits=[2.0, 0.5, 0.1])
        state = make_state()
        state = clf.run(state)
        assert state.primary_model_output.label == "positive"

    def test_run_appends_history_event(self):
        clf = _make_classifier(logits=[2.0, 0.5, 0.1])
        state = make_state()
        state = clf.run(state)
        assert len(state.history) == 1

    def test_run_history_component_name(self):
        clf = _make_classifier(logits=[2.0, 0.5, 0.1])
        state = make_state()
        clf.run(state)
        assert state.history[0].component == "primary_classifier"

    def test_run_history_contains_label_and_confidence(self):
        clf = _make_classifier(logits=[2.0, 0.5, 0.1])
        state = make_state()
        clf.run(state)
        outputs = state.history[0].outputs
        assert "label" in outputs
        assert "confidence" in outputs
        assert "probabilities" in outputs

    def test_run_history_contains_model_checkpoint(self):
        clf = _make_classifier(logits=[2.0, 0.5, 0.1])
        state = make_state()
        clf.run(state)
        assert "model" in state.history[0].outputs

    def test_run_raises_when_labels_empty(self):
        clf = _make_classifier(logits=[2.0, 0.5, 0.1])
        state = make_state(labels=[])
        with pytest.raises(ValueError, match="labels cannot be empty"):
            clf.run(state)

    def test_run_validates_label_against_task_config(self):
        """Label masking in predict() prevents unknown model classes from being
        selected, so run() should NOT raise even when the model has an
        out-of-task class with the highest logit."""
        clf = _make_classifier(
            logits=[2.0, 0.5, 0.1],
            id2label={0: "UNKNOWN_LABEL", 1: "negative", 2: "neutral"},
            label_map={0: "UNKNOWN_LABEL", 1: "negative", 2: "neutral"},
        )
        state = make_state(labels=["positive", "negative", "neutral"])
        # UNKNOWN_LABEL (index 0, highest logit) is masked out because it is
        # not in task_labels; the classifier must still produce a valid label.
        result = clf.run(state)
        assert result.primary_model_output.label in ("positive", "negative", "neutral")

    def test_run_probabilities_cover_task_labels(self):
        clf = _make_classifier(logits=[2.0, 0.5, 0.1])
        state = make_state()
        clf.run(state)
        probs = state.primary_model_output.probabilities
        for label in LABELS:
            assert label in probs

    def test_run_probabilities_sum_to_one(self):
        clf = _make_classifier(logits=[2.0, 0.5, 0.1])
        state = make_state()
        clf.run(state)
        probs = state.primary_model_output.probabilities
        assert abs(sum(probs.values()) - 1.0) < 1e-4

    def test_run_interface_compatible_with_orchestrator(self):
        """Verify run() signature: accepts PipelineState, returns PipelineState."""
        clf = _make_classifier(logits=[2.0, 0.5, 0.1])
        state = make_state()
        result = clf.run(state)
        assert isinstance(result, PipelineState)


# ---------------------------------------------------------------------------
# __repr__ test
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_checkpoint(self):
        _make_transformers_stub([1.0], {0: "positive"})
        from src.models.primary_transformer_classifier import PrimaryTransformerClassifier
        clf = PrimaryTransformerClassifier(checkpoint="my-model", label_map={0: "positive"})
        assert "my-model" in repr(clf)

    def test_repr_shows_not_loaded_before_load(self):
        _make_transformers_stub([1.0], {0: "positive"})
        from src.models.primary_transformer_classifier import PrimaryTransformerClassifier
        clf = PrimaryTransformerClassifier(checkpoint="my-model", label_map={0: "positive"})
        assert "not loaded" in repr(clf)

    def test_repr_shows_loaded_after_load(self):
        clf = _make_classifier(logits=[1.0, 0.0, 0.0])
        assert "loaded" in repr(clf)
        assert "not loaded" not in repr(clf)
