"""ner_conll_loader.py — Load Sabty-style CoNLL Arabic-English CS NER files.

File format (one "token TAG" per line, blank line between sentences):

    Cairo I-LOC
    ، O
    ...

The corpus uses the IO scheme (only ``I-`` prefixes) with four entity types:
PERS, LOC, ORG, MISC. This loader parses sentences and exposes helpers to
normalise tags to **type level** ({O, PERS, LOC, ORG, MISC}) so predictions from
a model using a different scheme (e.g. XLM-R's IOB2 ``B-PER``/``I-PER`` with no
MISC) can be compared fairly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

#: Canonical entity types in this corpus (macro-F1 averages over these, not O).
ENTITY_TYPES = ["PERS", "LOC", "ORG", "MISC"]
TYPE_LABELS = ["O"] + ENTITY_TYPES

# Map assorted model/gold surface tags onto this corpus's type names.
_TYPE_ALIASES: Dict[str, str] = {
    "PERS": "PERS", "PER": "PERS", "PERSON": "PERS",
    "LOC": "LOC", "LOCATION": "LOC", "GPE": "LOC",
    "ORG": "ORG", "ORGANISATION": "ORG", "ORGANIZATION": "ORG",
    "MISC": "MISC", "MISCELLANEOUS": "MISC",
}


def tag_to_type(tag: str) -> str:
    """Reduce any BIO/IO tag to its entity type, or 'O'.

    ``I-PERS`` -> ``PERS``; ``B-PER`` -> ``PERS``; ``O`` / unknown -> ``O``.
    """
    if not tag or tag == "O" or "-" not in tag:
        # bare type (no prefix) is possible too
        return _TYPE_ALIASES.get(tag.upper(), "O") if tag and tag != "O" else "O"
    etype = tag.split("-", 1)[1].upper()
    return _TYPE_ALIASES.get(etype, "O")


def load_conll(path: str | Path) -> List[Dict]:
    """Parse a CoNLL file into a list of ``{id, text, tokens, tags}`` dicts.

    ``tags`` are kept verbatim (e.g. ``I-PERS``). Use :func:`tag_to_type` to
    reduce them to type level for scheme-agnostic evaluation.
    """
    path = Path(path)
    sentences: List[Dict] = []
    tokens: List[str] = []
    tags: List[str] = []

    def _flush():
        if tokens:
            sentences.append({
                "id": f"{path.stem}_{len(sentences):05d}",
                "text": " ".join(tokens),
                "tokens": list(tokens),
                "tags": list(tags),
            })

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                _flush()
                tokens, tags = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            tokens.append(parts[0])
            tags.append(parts[-1])
    _flush()
    return sentences


def to_type_dataset(sentences: List[Dict]) -> List[Dict]:
    """Return a copy of *sentences* with tags reduced to type level."""
    out = []
    for s in sentences:
        out.append({
            **s,
            "tags": [tag_to_type(t) for t in s["tags"]],
        })
    return out
