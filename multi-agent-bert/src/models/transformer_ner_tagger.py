"""transformer_ner_tagger.py — Real token-classification (NER) tagger.

A sequence-labeling counterpart to
:class:`~src.models.primary_transformer_classifier.PrimaryTransformerClassifier`.
Where that class attaches a *sequence-classification* head (one label per
sentence, used by sentiment / topic), this class attaches a
*token-classification* head (one BIO tag per token) via
``AutoModelForTokenClassification`` — the standard way to do NER on an XLM-R /
BERT backbone.

It is used **only** on the NER path and never touches the classification path.

Default model
-------------
``Davlan/xlm-roberta-base-ner-hrl`` — XLM-RoBERTa base fine-tuned for NER on 10
high-resource languages incl. **Arabic and English**, with label set
``O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC`` (matches the synthetic
``data/dev_dummy_ner.jsonl`` exactly).

Offline / corporate-proxy note
-------------------------------
``checkpoint`` may be a Hub id **or a local directory**.  If Hub downloads are
blocked by an SSL proxy, download the model once elsewhere and pass the local
folder path.  You can also set ``HF_HUB_OFFLINE=1`` (and ``TRANSFORMERS_OFFLINE=1``)
to force local-only loading, or configure ``HTTPS_PROXY`` / ``REQUESTS_CA_BUNDLE``
for the proxy.

Sub-word alignment
------------------
Input is **pre-tokenised** (a list of word tokens).  We tokenise with
``is_split_into_words=True`` and map model sub-word predictions back to words
via ``word_ids()``, taking the **first sub-word** of each word as that word's
tag (the conventional choice).  ``torch`` / ``transformers`` are imported lazily
inside methods so importing this module stays cheap.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.state.schema import (
    AgentOutput,
    ModelOutput,
    PipelineState,
    SequenceLabelingOutput,
    TokenTag,
)

log = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT = "Davlan/xlm-roberta-base-ner-hrl"
_SKIP_NOTE = "TransformerNERTagger: task_type is not sequence_labeling — skipped."


class TransformerNERTagger:
    """HuggingFace transformer-backed BIO tagger for NER.

    This is the **NER primary model** — the sequence-labeling counterpart of
    :class:`~src.models.primary_transformer_classifier.PrimaryTransformerClassifier`.
    It exposes the same ``run(state) -> state`` entry point so the orchestrator
    can run it in the primary position of the NER path, with the heuristic NER
    agents acting as downstream specialists (mirroring the classification path:
    primary model → specialist agents → consensus).

    Parameters
    ----------
    checkpoint:
        HuggingFace model id (default ``Davlan/xlm-roberta-base-ner-hrl``) or a
        local directory containing the model + tokenizer files.
    label_map:
        Optional ``{model_class_index: tag_string}`` override.  When ``None`` the
        model's own ``config.id2label`` is used (present on Davlan NER models).
    device:
        ``"cpu"``, ``"cuda"``, ``"cuda:0"`` … Defaults to ``"cpu"``.
    max_length:
        Tokenizer max sequence length (default ``512``).
    name:
        Component name for logging.
    logger:
        Optional pre-configured logger.
    """

    def __init__(
        self,
        checkpoint: Union[str, Path] = _DEFAULT_CHECKPOINT,
        label_map: Optional[Dict[int, str]] = None,
        device: str = "cpu",
        max_length: int = 512,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        tag_normalizer=None,
    ) -> None:
        self.checkpoint = str(checkpoint)
        self.label_map = label_map
        self.device = device
        self.max_length = max_length
        self.name = name or "TransformerNERTagger"
        self.logger = logger or log
        # Optional callable applied to each raw model tag before it is checked
        # against the task labels — e.g. to reconcile label schemes/names
        # ("B-PER" -> "PERS"). None keeps the model's own tags. Used only to
        # match a specific corpus's label set; classification is unaffected.
        self.tag_normalizer = tag_normalizer

        self._tokenizer = None
        self._model = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: Union[str, Path] = _DEFAULT_CHECKPOINT,
        label_map: Optional[Dict[int, str]] = None,
        device: str = "cpu",
        max_length: int = 512,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        tag_normalizer=None,
    ) -> "TransformerNERTagger":
        """Construct **and** load the model in one call."""
        instance = cls(
            checkpoint=checkpoint,
            label_map=label_map,
            device=device,
            max_length=max_length,
            name=name,
            logger=logger,
            tag_normalizer=tag_normalizer,
        )
        instance.load()
        return instance

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load tokenizer + token-classification model. Idempotent."""
        if self._loaded:
            self.logger.debug("%s: model already loaded; skipping.", self.name)
            return

        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "The 'transformers' package is required for TransformerNERTagger. "
                "Install it with: pip install transformers"
            ) from exc

        try:
            import torch  # noqa: F401 (presence check)
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "The 'torch' package is required for TransformerNERTagger. "
                "Install it with: pip install torch"
            ) from exc

        self.logger.info("%s: loading tokenizer from '%s'", self.name, self.checkpoint)
        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        self.logger.info(
            "%s: loading token-classification model from '%s' → device='%s'",
            self.name, self.checkpoint, self.device,
        )
        if self.device == "auto":
            self._model = AutoModelForTokenClassification.from_pretrained(
                self.checkpoint, device_map="auto"
            )
        else:
            self._model = AutoModelForTokenClassification.from_pretrained(self.checkpoint)
            self._model.to(self.device)

        self._model.eval()

        # Derive the label map from the model config when not supplied.
        if self.label_map is None and hasattr(self._model.config, "id2label"):
            self.label_map = {int(k): v for k, v in self._model.config.id2label.items()}
            self.logger.debug(
                "%s: using id2label from model config: %s", self.name, self.label_map
            )

        self._loaded = True
        self.logger.info("%s: model loaded successfully.", self.name)

    # ------------------------------------------------------------------
    # Label normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_tag(raw_tag: str, valid_tags: set) -> str:
        """Map a model tag string onto the task's tag set, else ``"O"``.

        Handles common formatting differences (e.g. ``B-PER`` vs ``B-per`` vs
        ``PER``).  Any tag that cannot be matched falls back to ``"O"``.
        """
        if raw_tag in valid_tags:
            return raw_tag
        # Try upper-casing the entity type after the BIO prefix.
        if "-" in raw_tag:
            prefix, etype = raw_tag.split("-", 1)
            candidate = f"{prefix.upper()}-{etype.upper()}"
            if candidate in valid_tags:
                return candidate
        # Bare entity type (no BIO prefix) → treat as a B- tag if valid.
        candidate_b = f"B-{raw_tag.upper()}"
        if candidate_b in valid_tags:
            return candidate_b
        return "O"

    # ------------------------------------------------------------------
    # Sub-word → word alignment (pure, unit-testable)
    # ------------------------------------------------------------------

    @staticmethod
    def _align_first_subword(
        word_ids: List[Optional[int]],
        pred_ids: List[int],
        probs: List[float],
        n_words: int,
    ) -> List[Tuple[int, float]]:
        """Reduce sub-word predictions to one ``(label_id, prob)`` per word.

        Takes the **first** sub-word of each word (the first row whose
        ``word_ids`` equals that word index).  Words with no sub-word (should
        not happen for real tokenisers) default to ``(0, 0.0)`` — caller maps
        label id 0 through the label map like any other.

        Parameters
        ----------
        word_ids:
            Per-sub-word word index (``None`` for special tokens), from
            ``BatchEncoding.word_ids()``.
        pred_ids:
            Per-sub-word arg-max label id.
        probs:
            Per-sub-word max softmax probability.
        n_words:
            Number of input words (length of the output list).
        """
        out: List[Tuple[int, float]] = [(0, 0.0)] * n_words
        seen = [False] * n_words
        for sub_idx, w in enumerate(word_ids):
            if w is None or w >= n_words or seen[w]:
                continue
            seen[w] = True
            out[w] = (pred_ids[sub_idx], probs[sub_idx])
        return out

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------

    def tag(
        self,
        tokens: List[str],
        task_labels: Optional[List[str]] = None,
    ) -> SequenceLabelingOutput:
        """Tag a pre-tokenised sentence, returning one BIO tag per input token.

        Parameters
        ----------
        tokens:
            Pre-tokenised word list.
        task_labels:
            The task's valid tag set.  Model tags not in this set are mapped to
            ``"O"``.  When ``None``, the model's own tags are used verbatim.

        Returns
        -------
        SequenceLabelingOutput
            ``tags`` has exactly ``len(tokens)`` entries, in order.
        """
        if not self._loaded:
            raise RuntimeError(
                f"{self.name}: model not loaded. Call load() or use from_pretrained()."
            )

        if not tokens:
            return SequenceLabelingOutput(tags=[], notes=f"{self.name}: empty input.")

        import torch
        import torch.nn.functional as F

        encoding = self._tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = encoding.word_ids(batch_index=0)

        model_inputs = dict(encoding)
        if self.device != "auto":
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            logits = self._model(**model_inputs).logits  # [1, seq_len, num_labels]

        logits = logits.squeeze(0)              # [seq_len, num_labels]
        probs = F.softmax(logits, dim=-1)       # [seq_len, num_labels]
        pred_ids = probs.argmax(dim=-1).tolist()
        max_probs = probs.max(dim=-1).values.tolist()

        aligned = self._align_first_subword(word_ids, pred_ids, max_probs, len(tokens))

        valid_tags = set(task_labels) if task_labels else None
        tagged: List[TokenTag] = []
        for token, (label_id, prob) in zip(tokens, aligned):
            raw_tag = (self.label_map or {}).get(label_id, "O")
            if self.tag_normalizer is not None:
                raw_tag = self.tag_normalizer(raw_tag)
            tag = self._normalise_tag(raw_tag, valid_tags) if valid_tags else raw_tag
            tagged.append(TokenTag(token=token, tag=tag, confidence=round(float(prob), 6)))

        return SequenceLabelingOutput(
            tags=tagged,
            notes=f"{self.name}: tagged {len(tagged)} token(s) via {self.checkpoint}.",
            features={"checkpoint": self.checkpoint, "backbone": "transformer_token_classification"},
        )

    # ------------------------------------------------------------------
    # Orchestrator-compatible entry point (NER primary stage)
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """Tag ``state``'s tokens and write the result to ``state.ner_model_output``.

        Same ``run(state) -> state`` contract as the classification primary and
        the heuristic NER agents, so the orchestrator can slot it into the NER
        path's primary position without special-casing.  On non-NER tasks it is
        a no-op (records a skip note), so accidentally wiring it into a
        classification run cannot corrupt that path.
        """
        if state.task_config.task_type != "sequence_labeling":
            self.logger.debug(_SKIP_NOTE)
            state.append_history(component=self.name, summary=_SKIP_NOTE)
            return state

        if not state.task_config.labels:
            raise ValueError(f"{self.name}: state.task_config.labels is empty.")

        tokens: List[str] = state.extras.get("tokens") or state.input_text.split()
        seq_out = self.tag(tokens, task_labels=state.task_config.labels)

        state.ner_model_output = AgentOutput(
            agent_name=self.name,
            model_output=ModelOutput(),   # intentionally empty for NER
            sequence_output=seq_out,
            notes=seq_out.notes,
        )

        tag_summary = [f"{tt.token}:{tt.tag}" for tt in seq_out.tags]
        state.append_history(
            component="ner_primary_model",
            summary=f"NER primary tagging complete: {len(seq_out.tags)} token(s) via {self.checkpoint}.",
            outputs={"tags": tag_summary, "model": self.checkpoint, "device": self.device},
        )
        self.logger.debug(
            "%s: tagged %d token(s) via %s.", self.name, len(seq_out.tags), self.checkpoint
        )
        return state
