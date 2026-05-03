"""build_keyword_map.py — Extract discriminative keywords per label from a
labeled JSONL dataset and write a structured YAML keyword map.

Background
----------
The BERT multi-agent paper (SwitchLingua) defines per-label keyword and rule
maps for LexicalAgent and LogicAgent, but does not specify exactly how those
maps were constructed from data.  This script operationalises that missing
step: given a labeled dataset it computes statistically discriminative terms
for each label and emits them in the YAML format consumed by
``evaluate_pipeline.build_agent_knowledge_maps()``.

Two scoring methods are supported (``--method``):

* ``chi2``  (default) — chi-square statistic between each term and each label,
  evaluated as a one-vs-rest binary classification problem.  Selects terms
  whose presence is most non-random for a given class.

* ``tfidf`` — average TF-IDF weight of a term across documents that belong to
  the label.  Selects terms that are both frequent within the class and rare
  globally.

Both methods are implemented with stdlib only (no scikit-learn / scipy).

Output YAML structure (per label)
----------------------------------
    <label>:
      keywords_l1: [...]   # Arabic-script terms (Unicode Arabic block)
      keywords_l2: [...]   # Latin-script terms
      regex_rules:
        - \\b(term1|term2|...)\\b   # Latin-script rule
        - (term1|term2|...)         # Arabic-script rule (no word-boundary)

Usage
-----
    python scripts/build_keyword_map.py \\
        --input data/dev_dummy.jsonl \\
        --output config/keyword_map.yaml \\
        --top_k 10 \\
        --min_df 2

CLI flags
---------
    --input     PATH     JSONL file with "text" and "label" fields (required)
    --output    PATH     Output YAML file (default: keyword_map.yaml)
    --top_k     INT      Max keywords per label per script type (default: 10)
    --min_df    INT      Min document frequency for a term to be considered (default: 2)
    --method    STR      Scoring method: chi2 or tfidf (default: chi2)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml  # PyYAML — already in requirements.txt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_keyword_map")

# ---------------------------------------------------------------------------
# Arabic script detection
# Arabic: U+0600–U+06FF, Arabic Supplement: U+0750–U+077F,
# Arabic Presentation Forms-A: U+FB50–U+FDFF,
# Arabic Presentation Forms-B: U+FE70–U+FEFF
# ---------------------------------------------------------------------------
_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)


def _is_arabic_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _ARABIC_RANGES)


def _is_arabic_term(term: str) -> bool:
    """Return True if the term contains at least one Arabic-script character."""
    return any(_is_arabic_char(ch) for ch in term)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def tokenize(text: str) -> List[str]:
    """Tokenize mixed Arabic-English text.

    Strategy:
    - Split on whitespace.
    - Strip leading/trailing punctuation from each token.
    - Lowercase Latin-script tokens; preserve Arabic-script casing (irrelevant
      for Arabic but keeps the token intact).
    - Discard single-character tokens and purely numeric tokens.
    """
    tokens: List[str] = []
    for raw in text.split():
        # Strip non-word, non-Arabic characters from edges
        tok = _PUNCT_RE.sub(" ", raw).strip()
        for part in tok.split():
            part = part.strip()
            if len(part) < 2:
                continue
            if part.isdigit():
                continue
            # Lowercase only if purely Latin (preserve Arabic)
            if not _is_arabic_term(part):
                part = part.lower()
            tokens.append(part)
    return tokens


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, str]]:
    """Load JSONL; each record must have 'text' and 'label'."""
    records: List[Dict[str, str]] = []
    skipped = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Line %d — JSON error: %s", lineno, exc)
                skipped += 1
                continue
            if "text" not in obj or "label" not in obj:
                log.warning("Line %d — missing 'text' or 'label', skipping.", lineno)
                skipped += 1
                continue
            records.append({"text": str(obj["text"]), "label": str(obj["label"])})
    if not records:
        raise ValueError(f"No valid records found in {path}")
    log.info("Loaded %d records (%d skipped).", len(records), skipped)
    return records


# ---------------------------------------------------------------------------
# Term × label statistics
# ---------------------------------------------------------------------------

def build_doc_term_matrix(
    records: List[Dict[str, str]],
    min_df: int,
) -> Tuple[List[str], Dict[str, List[int]]]:
    """Return vocabulary and per-label document-presence lists.

    Returns
    -------
    vocab : sorted list of terms that pass min_df filter
    label_doc_ids : mapping label → list of record indices
    """
    # Build vocabulary with document frequencies
    doc_freq: Counter[str] = Counter()
    tokenized: List[List[str]] = []
    for rec in records:
        toks = list(set(tokenize(rec["text"])))  # unique terms per doc
        tokenized.append(toks)
        doc_freq.update(toks)

    vocab = sorted(t for t, df in doc_freq.items() if df >= min_df)
    vocab_set = set(vocab)
    log.info("Vocabulary: %d terms (min_df=%d).", len(vocab), min_df)

    # Build per-label doc-id lists
    label_doc_ids: Dict[str, List[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        label_doc_ids[rec["label"]].append(idx)

    return vocab, tokenized, doc_freq, label_doc_ids


# ---------------------------------------------------------------------------
# Scoring — chi-square (one-vs-rest per label)
# ---------------------------------------------------------------------------

def chi2_scores(
    vocab: List[str],
    tokenized: List[List[str]],
    label_doc_ids: Dict[str, List[int]],
    target_label: str,
) -> Dict[str, float]:
    """Compute chi-square score for each term vs target_label (one-vs-rest).

    chi2 = N * (AD - BC)^2 / ((A+B)(C+D)(A+C)(B+D))
    where for each term t:
      A = in-class docs containing t
      B = out-of-class docs containing t
      C = in-class docs NOT containing t
      D = out-of-class docs NOT containing t
    """
    N = len(tokenized)
    in_class = set(label_doc_ids.get(target_label, []))
    n_in = len(in_class)
    n_out = N - n_in

    scores: Dict[str, float] = {}
    for term in vocab:
        # Count co-occurrences
        a = sum(1 for idx in in_class if term in tokenized[idx])
        b = sum(1 for idx in range(N) if idx not in in_class and term in tokenized[idx])
        c = n_in - a
        d = n_out - b

        denom = (a + b) * (c + d) * (a + c) * (b + d)
        if denom == 0:
            scores[term] = 0.0
        else:
            scores[term] = N * (a * d - b * c) ** 2 / denom

    return scores


# ---------------------------------------------------------------------------
# Scoring — TF-IDF (average per-class)
# ---------------------------------------------------------------------------

def tfidf_scores(
    vocab: List[str],
    tokenized: List[List[str]],
    doc_freq: Counter,
    label_doc_ids: Dict[str, List[int]],
    target_label: str,
) -> Dict[str, float]:
    """Average TF-IDF of each term over documents belonging to target_label."""
    N = len(tokenized)
    in_class_ids = label_doc_ids.get(target_label, [])
    if not in_class_ids:
        return {t: 0.0 for t in vocab}

    idf: Dict[str, float] = {
        t: math.log((N + 1) / (doc_freq[t] + 1)) + 1.0 for t in vocab
    }

    scores: Dict[str, float] = {t: 0.0 for t in vocab}
    for idx in in_class_ids:
        tok_counts = Counter(tokenized[idx])
        total = sum(tok_counts.values()) or 1
        for term in vocab:
            tf = tok_counts.get(term, 0) / total
            scores[term] += tf * idf[term]

    # Average over in-class docs
    n = len(in_class_ids)
    return {t: v / n for t, v in scores.items()}


# ---------------------------------------------------------------------------
# Build keyword map
# ---------------------------------------------------------------------------

def _top_terms(scores: Dict[str, float], top_k: int) -> List[str]:
    """Return top_k terms sorted by descending score, ties broken alphabetically."""
    return [t for t, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]]


def _build_regex_rule(terms: List[str], arabic: bool) -> str | None:
    """Build a single alternation regex from a list of terms.

    For Latin terms: wrap in \\b word-boundaries.
    For Arabic terms: plain alternation (Arabic script has no word-boundary concept
    in standard regex engines without Unicode mode).
    """
    if not terms:
        return None
    joined = "|".join(re.escape(t) for t in terms)
    if arabic:
        return f"({joined})"
    else:
        return f"\\b({joined})\\b"


def build_keyword_map(
    records: List[Dict[str, str]],
    top_k: int,
    min_df: int,
    method: str,
) -> Dict[str, Dict]:
    """Core extraction pipeline.

    Returns a nested dict keyed by label with sub-keys
    keywords_l1 (Arabic), keywords_l2 (Latin), regex_rules.
    """
    vocab, tokenized, doc_freq, label_doc_ids = build_doc_term_matrix(records, min_df)
    labels = sorted(label_doc_ids.keys())
    log.info("Labels: %s", labels)

    result: Dict[str, Dict] = {}

    for label in labels:
        log.info("Scoring label '%s' (method=%s) …", label, method)

        if method == "tfidf":
            scores = tfidf_scores(vocab, tokenized, doc_freq, label_doc_ids, label)
        else:  # chi2 (default)
            scores = chi2_scores(vocab, tokenized, label_doc_ids, label)

        top = _top_terms(scores, top_k * 4)  # over-select before splitting by script

        arabic_terms = [t for t in top if _is_arabic_term(t)][:top_k]
        latin_terms  = [t for t in top if not _is_arabic_term(t)][:top_k]

        regex_rules: List[str] = []
        latin_rule  = _build_regex_rule(latin_terms,  arabic=False)
        arabic_rule = _build_regex_rule(arabic_terms, arabic=True)
        if latin_rule:
            regex_rules.append(latin_rule)
        if arabic_rule:
            regex_rules.append(arabic_rule)

        result[label] = {
            "keywords_l1": arabic_terms,   # Arabic-script
            "keywords_l2": latin_terms,    # Latin-script
            "regex_rules": regex_rules,
        }

        log.info(
            "  '%s' → %d Arabic, %d Latin keywords",
            label, len(arabic_terms), len(latin_terms),
        )

    return result


# ---------------------------------------------------------------------------
# YAML serialisation
# ---------------------------------------------------------------------------

def _dump_yaml(data: Dict, path: Path) -> None:
    """Write keyword map to YAML with readable formatting."""
    # Use a custom representer so lists stay inline for short lists.
    class _InlineLists(yaml.Dumper):
        pass

    def _represent_list(dumper, lst):
        if all(isinstance(x, str) and len(x) < 40 for x in lst):
            return dumper.represent_sequence("tag:yaml.org,2002:seq", lst, flow_style=True)
        return dumper.represent_sequence("tag:yaml.org,2002:seq", lst, flow_style=False)

    _InlineLists.add_representer(list, _represent_list)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            "# Auto-generated keyword map — produced by scripts/build_keyword_map.py\n"
            "# Operationalises the paper's unspecified keyword-map construction step.\n"
            "# keywords_l1 = Arabic-script terms\n"
            "# keywords_l2 = Latin-script terms\n"
            "# regex_rules  = alternation patterns for LogicAgent\n\n"
        )
        yaml.dump(
            data,
            fh,
            Dumper=_InlineLists,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=True,
        )
    log.info("Wrote keyword map to '%s'.", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract discriminative keywords per label from a labeled JSONL dataset "
            "and write a structured YAML keyword map for LexicalAgent / LogicAgent. "
            "Operationalises the paper's unspecified keyword-map construction step."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="JSONL file with 'text' and 'label' fields.",
    )
    parser.add_argument(
        "--output",
        default="keyword_map.yaml",
        metavar="PATH",
        help="Output YAML file (default: keyword_map.yaml).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        metavar="INT",
        help="Max keywords per label per script type (default: 10).",
    )
    parser.add_argument(
        "--min_df",
        type=int,
        default=2,
        metavar="INT",
        help="Minimum document frequency for a term to be included (default: 2).",
    )
    parser.add_argument(
        "--method",
        default="chi2",
        choices=["chi2", "tfidf"],
        metavar="METHOD",
        help="Scoring method: chi2 (default) or tfidf.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        return 1

    try:
        records = load_jsonl(input_path)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    keyword_map = build_keyword_map(
        records,
        top_k=args.top_k,
        min_df=args.min_df,
        method=args.method,
    )

    _dump_yaml(keyword_map, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
