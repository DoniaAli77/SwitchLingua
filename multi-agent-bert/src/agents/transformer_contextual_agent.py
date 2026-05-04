"""Non-LLM contextual agent for paper_style pipeline mode.

Approximates the "contextual agent" described in the reference BERT multi-agent
framework (BERT / RoBERTa / XLNet) without requiring a fine-tuned model or an
LLM API call.

Two operating modes
-------------------
``tfidf`` (default)
    Deterministic TF-IDF cosine-similarity between the input text and each label
    description string.  Requires only the Python standard library.  Suitable
    for unit tests, offline evaluation, and any environment without GPU/internet.

``embedding``
    Mean-pooled sentence embeddings from a HuggingFace transformer model.
    Requires ``transformers`` and ``torch``.  If either package is absent, or if
    model loading/inference fails for any reason, the agent logs a warning and
    silently falls back to ``tfidf``.  This means the agent always produces a
    result regardless of the runtime environment.

Input / output contract
-----------------------
* Reads  : ``state.input_text``, ``state.task_config.labels``,
           ``state.task_config.label_descriptions``
* Writes : ``state.contextual_output`` (an :class:`~src.state.schema.AgentOutput`)
           with ``model_output.label``, ``model_output.confidence``,
           ``model_output.probabilities``, ``notes``, and
           ``features["similarity_scores"]``.

No changes to :class:`~src.state.schema.PipelineState` or
:class:`~src.state.schema.AgentOutput` are required.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from src.agents.base_agent import BaseAgent
from src.state.schema import AgentOutput, ModelOutput, PipelineState

_AGENT_NAME = "TransformerContextualAgent"
_NO_MATCH_NOTE = "No similarity signal; uniform fallback applied."
_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------------------------
# Private helpers — TF-IDF
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    r"""Lowercase word-boundary tokenization.

    Uses ``\b\w+\b`` so Arabic words (Unicode \w) are captured correctly.
    """
    return re.findall(r"\b\w+\b", text.lower())


def _tfidf_similarity(
    input_text: str,
    label_descriptions: Dict[str, str],
) -> Dict[str, float]:
    """Return raw TF-IDF cosine similarity between *input_text* and each label description.

    Algorithm
    ---------
    1. Treat each label description as one document; build IDF over that corpus.
    2. Compute a TF-IDF vector for the input text (restricted to known vocabulary).
    3. Compute a TF-IDF vector for each description.
    4. Return the cosine similarity between the input vector and each description
       vector.  Zero-norm vectors yield a similarity of 0.0.
    """
    docs: Dict[str, List[str]] = {
        label: _tokenize(desc) for label, desc in label_descriptions.items()
    }
    n_docs = max(len(docs), 1)

    # Document frequency per term across label-description corpus.
    df: Counter[str] = Counter()
    for tokens in docs.values():
        for term in set(tokens):
            df[term] += 1

    # Smooth IDF:  log((1+N)/(1+df)) + 1
    idf: Dict[str, float] = {
        term: math.log((1.0 + n_docs) / (1.0 + cnt)) + 1.0
        for term, cnt in df.items()
    }

    # TF-IDF vector for input text (only vocabulary terms contribute).
    input_tokens = _tokenize(input_text)
    input_tf = Counter(input_tokens)
    input_len = max(len(input_tokens), 1)
    input_vec: Dict[str, float] = {
        term: (cnt / input_len) * idf[term]
        for term, cnt in input_tf.items()
        if term in idf
    }
    input_norm = math.sqrt(sum(v ** 2 for v in input_vec.values()))

    scores: Dict[str, float] = {}
    for label, tokens in docs.items():
        doc_tf = Counter(tokens)
        doc_len = max(len(tokens), 1)
        doc_vec: Dict[str, float] = {
            term: (cnt / doc_len) * idf.get(term, 0.0)
            for term, cnt in doc_tf.items()
        }
        doc_norm = math.sqrt(sum(v ** 2 for v in doc_vec.values()))

        if input_norm > 0 and doc_norm > 0:
            dot = sum(input_vec.get(t, 0.0) * doc_vec.get(t, 0.0) for t in doc_vec)
            scores[label] = dot / (input_norm * doc_norm)
        else:
            scores[label] = 0.0

    return scores


# ---------------------------------------------------------------------------
# Private helpers — embedding (optional)
# ---------------------------------------------------------------------------

def _embedding_similarity(
    input_text: str,
    label_descriptions: Dict[str, str],
    model_name: str,
) -> Optional[Dict[str, float]]:
    """Compute cosine similarity via mean-pooled HuggingFace embeddings.

    Returns ``None`` if ``transformers`` / ``torch`` are unavailable or if
    model loading or inference raises any exception.
    """
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
    except ImportError:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        labels = list(label_descriptions.keys())
        texts = [input_text] + [label_descriptions[lbl] for lbl in labels]

        encoding = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128,
        )

        with torch.no_grad():
            output = model(**encoding)

        # Mean pooling over non-padding tokens.
        token_embeds = output.last_hidden_state          # (n, seq_len, hidden)
        mask = encoding["attention_mask"].unsqueeze(-1).float()
        mean_embeds = (token_embeds * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        # L2-normalise.
        norms = mean_embeds.norm(dim=1, keepdim=True).clamp(min=1e-9)
        embeds = mean_embeds / norms  # (n, hidden)

        input_embed = embeds[0]
        label_embeds = embeds[1:]

        scores: Dict[str, float] = {
            label: float((input_embed * label_embeds[i]).sum().clamp(min=0.0).item())
            for i, label in enumerate(labels)
        }
        return scores

    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------

def _normalize_scores(
    scores: Dict[str, float],
) -> Tuple[Dict[str, float], str]:
    """Normalise raw similarity scores to a probability-like distribution.

    Returns ``(probabilities, fallback_note)``.  ``fallback_note`` is non-empty
    only when the uniform fallback is applied (all scores are zero).
    """
    total = sum(scores.values())
    if total <= 0.0:
        n = max(len(scores), 1)
        uniform = round(1.0 / n, 6)
        return {label: uniform for label in scores}, _NO_MATCH_NOTE
    return {label: round(v / total, 6) for label, v in scores.items()}, ""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TransformerContextualAgent(BaseAgent[PipelineState]):
    """Deterministic, non-LLM contextual agent for paper_style mode.

    Scores each label by computing the similarity between the input text and
    the corresponding label description.  No HTTP requests or model downloads
    are needed when using the default ``tfidf`` mode.

    Parameters
    ----------
    mode:
        ``"tfidf"`` (default) — stdlib TF-IDF cosine similarity, always
        available and fully deterministic.
        ``"embedding"`` — mean-pooled HuggingFace transformer embeddings;
        silently falls back to ``"tfidf"`` if ``transformers`` / ``torch``
        are not installed or if model loading/inference fails.
    model_name:
        HuggingFace model identifier used in ``"embedding"`` mode.  Ignored
        in ``"tfidf"`` mode.  Defaults to a multilingual sentence-transformer.
    name:
        Optional agent name for logging and ``AgentOutput.agent_name``.
    logger:
        Optional pre-configured logger.
    """

    def __init__(
        self,
        mode: str = "tfidf",
        model_name: str = _DEFAULT_MODEL,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name=name or _AGENT_NAME, logger=logger)
        if mode not in ("tfidf", "embedding"):
            raise ValueError(
                f"TransformerContextualAgent: unknown mode {mode!r}. "
                "Allowed values: 'tfidf', 'embedding'."
            )
        self.mode = mode
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Validation hooks
    # ------------------------------------------------------------------

    def validate_before(self, state: PipelineState) -> None:
        if not state.input_text or not state.input_text.strip():
            raise ValueError(f"{self.name}: state.input_text is empty or blank.")
        if not state.task_config.labels:
            raise ValueError(f"{self.name}: state.task_config.labels is empty.")

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """Compute similarity scores and write result to ``state.contextual_output``."""
        text = state.input_text
        labels = state.task_config.labels
        descriptions = state.task_config.label_descriptions or {}

        # Fall back to the label name when no description is provided.
        effective_descs: Dict[str, str] = {
            label: descriptions.get(label, label) for label in labels
        }

        # --- Compute similarity -------------------------------------------
        effective_mode = self.mode
        raw_scores: Optional[Dict[str, float]] = None

        if effective_mode == "embedding":
            raw_scores = _embedding_similarity(text, effective_descs, self.model_name)
            if raw_scores is None:
                self.logger.warning(
                    "%s: embedding mode unavailable or failed; falling back to tfidf.",
                    self.name,
                )
                effective_mode = "tfidf"

        if raw_scores is None:  # primary tfidf path or embedding fallback
            raw_scores = _tfidf_similarity(text, effective_descs)

        # --- Normalise and select best label -----------------------------
        probs, fallback_note = _normalize_scores(raw_scores)
        best_label = max(labels, key=lambda lbl: probs.get(lbl, 0.0))
        confidence = round(probs.get(best_label, 0.0), 6)

        note_parts = [
            f"Mode: {effective_mode}.",
            f"Best match: '{best_label}' (similarity={confidence:.4f}).",
        ]
        if fallback_note:
            note_parts.append(fallback_note)

        notes = " ".join(note_parts)
        state.contextual_output = AgentOutput(
            agent_name=self.name,
            model_output=ModelOutput(
                label=best_label,
                confidence=confidence,
                probabilities=probs,
            ),
            notes=notes,
            features={
                "similarity_scores": raw_scores,
                "effective_mode": effective_mode,
            },
        )
        state.append_history(
            component=self.name,
            summary=(
                f"Non-finetuned contextual similarity ({effective_mode}): "
                f"label='{best_label}' confidence={confidence:.4f}."
            ),
            outputs={
                "label": best_label,
                "confidence": confidence,
                "probabilities": dict(probs),
                "effective_mode": effective_mode,
                "similarity_scores": {k: round(v, 6) for k, v in raw_scores.items()},
            },
        )
        return state
