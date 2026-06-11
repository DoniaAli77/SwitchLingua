"""Shared rendering of the optional primary-signal prompt block (Fix #3).

Turns the primary model's prediction into a compact context block that the LLM
classifier agents may receive when ``task_config.agents_use_primary_signal`` is
enabled. Task-generic: no label names are hardcoded; top-2 is by probability
value, never label order. Default usage renders the empty string, so the block is
absent unless a usable signal is supplied.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def build_primary_signal(primary_output: Any) -> Optional[Dict[str, Any]]:
    """Build a primary-signal dict from a ``ModelOutput``, or ``None``.

    Returns ``None`` when there is no usable primary label (so the agent runs
    blind, as before). ``top2`` is the two highest-probability ``(label, prob)``
    pairs, sorted by probability descending.
    """
    if primary_output is None or getattr(primary_output, "label", None) is None:
        return None
    probs: Dict[str, float] = dict(getattr(primary_output, "probabilities", {}) or {})
    ranked: List[Tuple[str, float]] = sorted(
        probs.items(), key=lambda kv: kv[1], reverse=True
    )
    return {
        "label": primary_output.label,
        "confidence": getattr(primary_output, "confidence", None),
        "top2": ranked[:2],
        "distribution": probs,
    }


def render_primary_block(signal: Optional[Dict[str, Any]], analysis_kind: str) -> str:
    """Render the primary-signal block, or ``""`` when there is no usable signal.

    Parameters
    ----------
    signal:
        The dict from :func:`build_primary_signal`, or ``None`` (→ empty string).
    analysis_kind:
        The agent's own role word (e.g. ``"lexical"``, ``"logical"``,
        ``"contextual"``) — used only in the anti-anchoring wording, never a label.
    """
    if not signal or signal.get("label") is None:
        return ""

    conf = signal.get("confidence")
    conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"

    lines: List[str] = [
        f"PRIMARY MODEL SIGNAL (context only — you are an independent "
        f"{analysis_kind} adjudicator):",
        f"  predicted label  : {signal['label']}",
        f"  confidence       : {conf_str}",
    ]

    top2: List[Tuple[str, float]] = list(signal.get("top2") or [])
    dist: Dict[str, float] = dict(signal.get("distribution") or {})
    if top2:
        lines.append(
            "  top-2 labels     : "
            + ", ".join(f"{lbl} ({p:.2f})" for lbl, p in top2)
        )
    if dist:
        ordered = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        lines.append(
            "  full distribution: " + ", ".join(f"{lbl}={p:.2f}" for lbl, p in ordered)
        )
    if not top2 and not dist:
        lines.append("  (probability distribution unavailable)")

    # Anti-anchoring guidance.
    lines.append(
        f"The primary may be wrong — especially when its confidence is low or its "
        f"top-2 are close. Do your own {analysis_kind} analysis of the text FIRST. "
        f"Use this signal as context only; agree only if the text evidence supports "
        f"the primary's label, and choose a different allowed label if the evidence "
        f"points elsewhere. Do NOT simply copy the primary."
    )
    return "\n".join(lines)
