"""Real primary classifier backed by a Hugging Face transformer model.

Replaces :class:`~src.models.mock_primary_classifier.MockPrimaryClassifier`
in production while keeping the same ``run(state) -> state`` interface so the
:class:`~src.pipeline.orchestrator.PipelineOrchestrator` requires no changes.

Usage
-----
**Load from a local checkpoint or Hub model id:**

.. code-block:: python

    from src.models.primary_transformer_classifier import PrimaryTransformerClassifier

    clf = PrimaryTransformerClassifier.from_pretrained(
        checkpoint="distilbert-base-uncased-finetuned-sst-2-english",
        label_map={0: "negative", 1: "positive"},
        # optional — device / dtype / max_length overrides
    )

**Or construct then call ``load()`` manually:**

.. code-block:: python

    clf = PrimaryTransformerClassifier(
        checkpoint="path/to/local/model",
        label_map={0: "negative", 1: "positive"},
    )
    clf.load()

**Wire into the orchestrator:**

.. code-block:: python

    orchestrator = PipelineOrchestrator(
        primary_classifier=clf,   # same slot as MockPrimaryClassifier
        ...
    )

Label mapping
-------------
The model's integer class indices are mapped to task label strings via
``label_map``.  If ``label_map`` is omitted, the classifier falls back to
the model's own ``config.id2label`` mapping (present on most Hub models).
When both are absent, raw integer strings are used as label names.

The mapped labels are then **intersected with ``task_config.labels``** at
prediction time so only labels the task actually uses are reported.  Logits
for labels not in ``task_config.labels`` are excluded before softmax.

Thread safety
-------------
The model and tokenizer are loaded once in ``load()`` / ``from_pretrained()``
and are thereafter read-only.  ``run()`` is safe to call from multiple threads
provided the underlying PyTorch model is not being fine-tuned concurrently.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.state.schema import ModelOutput, PipelineState

log = logging.getLogger(__name__)

# Sentinel used in history when the model was loaded from a Hub id vs. local path.
_SOURCE_HUB = "huggingface_hub"
_SOURCE_LOCAL = "local_checkpoint"


class PrimaryTransformerClassifier:
    """HuggingFace transformer-backed primary classifier.

    Parameters
    ----------
    checkpoint:
        A HuggingFace model id (e.g. ``"distilbert-base-uncased-finetuned-sst-2-english"``)
        or a local directory path containing ``config.json``, ``pytorch_model.bin``
        (or ``model.safetensors``), and ``tokenizer_config.json``.
    label_map:
        Optional mapping of ``{model_class_index: task_label_string}``.
        When provided it takes priority over ``config.id2label``.
        Example: ``{0: "negative", 1: "positive"}``.
    device:
        PyTorch device string such as ``"cpu"``, ``"cuda"``, or ``"cuda:0"``.
        Defaults to ``"cpu"``.  Pass ``"auto"`` to let the ``device_map``
        logic in *Accelerate* choose automatically (requires ``accelerate``).
    max_length:
        Maximum token length passed to the tokenizer.  Defaults to ``512``.
    name:
        Component name used in history events.  Defaults to
        ``"PrimaryTransformerClassifier"``.
    logger:
        Optional pre-configured logger.
    """

    def __init__(
        self,
        checkpoint: Union[str, Path],
        label_map: Optional[Dict[int, str]] = None,
        device: str = "cpu",
        max_length: int = 512,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.checkpoint = str(checkpoint)
        self.label_map = label_map  # may be None until load() or from_pretrained()
        self.device = device
        self.max_length = max_length
        self.name = name or "PrimaryTransformerClassifier"
        self.logger = logger or log

        # Set after load() / from_pretrained()
        self._tokenizer = None
        self._model = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: Union[str, Path],
        label_map: Optional[Dict[int, str]] = None,
        device: str = "cpu",
        max_length: int = 512,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> "PrimaryTransformerClassifier":
        """Construct **and** load the model in one call.

        Parameters
        ----------
        checkpoint:
            HuggingFace model id or local checkpoint directory.
        label_map:
            Optional ``{class_index: label_string}`` mapping.
        device:
            PyTorch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``).
        max_length:
            Tokenizer maximum sequence length.
        name:
            Component name used in pipeline history events.
        logger:
            Optional pre-configured logger.

        Returns
        -------
        PrimaryTransformerClassifier
            A fully loaded, ready-to-use classifier instance.
        """
        instance = cls(
            checkpoint=checkpoint,
            label_map=label_map,
            device=device,
            max_length=max_length,
            name=name,
            logger=logger,
        )
        instance.load()
        return instance

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load tokenizer and model from ``self.checkpoint``.

        Safe to call multiple times — subsequent calls are no-ops.

        Raises
        ------
        ImportError
            If ``transformers`` is not installed.
        OSError
            If ``checkpoint`` does not exist on disk or cannot be fetched.
        """
        if self._loaded:
            self.logger.debug("%s: model already loaded; skipping.", self.name)
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required for PrimaryTransformerClassifier. "
                "Install it with: pip install transformers"
            ) from exc

        try:
            import torch  # noqa: F401 (presence check)
        except ImportError as exc:
            raise ImportError(
                "The 'torch' package is required for PrimaryTransformerClassifier. "
                "Install it with: pip install torch"
            ) from exc

        self.logger.info(
            "%s: loading tokenizer from '%s'", self.name, self.checkpoint
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        self.logger.info(
            "%s: loading model from '%s' → device='%s'",
            self.name, self.checkpoint, self.device,
        )
        if self.device == "auto":
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.checkpoint, device_map="auto"
            )
        else:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.checkpoint
            )
            self._model.to(self.device)

        self._model.eval()

        # Build label map from model config if the caller did not supply one.
        if self.label_map is None and hasattr(self._model.config, "id2label"):
            self.label_map = {
                int(k): v for k, v in self._model.config.id2label.items()
            }
            self.logger.debug(
                "%s: using id2label from model config: %s", self.name, self.label_map
            )

        self._loaded = True
        self.logger.info("%s: model loaded successfully.", self.name)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        text: str,
        task_labels: Optional[List[str]] = None,
    ) -> ModelOutput:
        """Run inference on a single text string.

        Parameters
        ----------
        text:
            The raw input text to classify.
        task_labels:
            When supplied, only logits whose mapped label is in this list
            are included in the softmax and probability output.  Logits for
            unlisted labels are masked out (set to ``-inf``) before softmax,
            so probabilities always sum to 1.0 over the task label set.
            Pass ``None`` to use all model output classes.

        Returns
        -------
        ModelOutput
            ``label``         — the predicted label string  
            ``confidence``    — softmax probability of the winning label  
            ``probabilities`` — full probability distribution over task labels  
            ``raw_text``      — the original input text (passed through)

        Raises
        ------
        RuntimeError
            If ``load()`` has not been called first.
        """
        if not self._loaded:
            raise RuntimeError(
                f"{self.name}: model not loaded. Call load() or use from_pretrained()."
            )

        import torch
        import torch.nn.functional as F

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )

        # Move input tensors to the same device as the model.
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits: torch.Tensor = self._model(**inputs).logits  # shape: [1, num_labels]

        logits = logits.squeeze(0)  # [num_labels]
        num_labels = logits.size(0)

        # Build index → label_string mapping for *all* model outputs.
        full_label_map: Dict[int, str] = {}
        for i in range(num_labels):
            if self.label_map and i in self.label_map:
                full_label_map[i] = self.label_map[i]
            else:
                full_label_map[i] = str(i)

        # Mask logits whose label is not in task_labels (if supplied).
        if task_labels is not None:
            task_label_set = set(task_labels)
            for i, lbl in full_label_map.items():
                if lbl not in task_label_set:
                    logits[i] = float("-inf")

        probs: torch.Tensor = F.softmax(logits, dim=-1)  # [num_labels]
        best_idx: int = int(probs.argmax().item())

        predicted_label: str = full_label_map[best_idx]
        confidence: float = round(float(probs[best_idx].item()), 6)

        # Build probability dict — only include indices with finite logits.
        probabilities: Dict[str, float] = {}
        for i, lbl in full_label_map.items():
            p = float(probs[i].item())
            if p > 0.0 or (task_labels is None or lbl in task_labels):
                probabilities[lbl] = round(p, 6)

        # If task_labels was given, restrict the dict to those labels only.
        if task_labels is not None:
            probabilities = {lbl: probabilities.get(lbl, 0.0) for lbl in task_labels}

        return ModelOutput(
            label=predicted_label,
            confidence=confidence,
            probabilities=probabilities,
            raw_text=text,
        )

    # ------------------------------------------------------------------
    # Orchestrator-compatible entry point
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """Classify ``state.input_text`` and write result to ``state.primary_model``.

        This method has the same signature and semantics as
        :meth:`~src.models.mock_primary_classifier.MockPrimaryClassifier.run`
        so the :class:`~src.pipeline.orchestrator.PipelineOrchestrator` can
        accept either classifier without modification.

        Raises
        ------
        ValueError
            If ``state.task_config.labels`` is empty.
        RuntimeError
            If the model has not been loaded.
        """
        labels: List[str] = state.task_config.labels
        if not labels:
            raise ValueError(
                f"{self.name}: state.task_config.labels cannot be empty."
            )

        output = self.predict(text=state.input_text, task_labels=labels)

        if output.label is None:
            raise ValueError(f"{self.name}: prediction returned no label.")  # pragma: no cover

        state.task_config.validate_label(output.label, field_name="primary_model.label")
        output.validate_labels(state.task_config, field_name="primary_model.label")

        state.primary_model = output

        source = (
            _SOURCE_LOCAL
            if Path(self.checkpoint).exists()
            else _SOURCE_HUB
        )
        state.append_history(
            component="primary_classifier",
            summary=(
                f"Predicted '{output.label}' with confidence "
                f"{output.confidence:.3f} (model='{self.checkpoint}')"
            ),
            outputs={
                "label": output.label,
                "confidence": output.confidence,
                "probabilities": dict(output.probabilities),
                "model": self.checkpoint,
                "source": source,
                "device": self.device,
                "max_length": self.max_length,
            },
        )
        return state

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        loaded = "loaded" if self._loaded else "not loaded"
        return (
            f"PrimaryTransformerClassifier("
            f"checkpoint={self.checkpoint!r}, "
            f"device={self.device!r}, "
            f"max_length={self.max_length}, "
            f"status={loaded})"
        )
