"""Shared rendering of the optional SEQUENTIAL agent-chain prompt block.

Supports the fair "same-agents, sequential order" ablation: the exact parallel
specialists (Lexical, Polarity, Contextual) are run in a fixed order, and each
later agent receives the earlier agents' conclusions as context. This block is
rendered **only** when ``task_config.sequential_chain`` is enabled; default usage
returns the empty string, so the parallel behaviour is byte-for-byte unchanged.

Deliberately agent-only: it lists prior SPECIALIST outputs (label / confidence /
short note) and never the primary model — so the sole difference from the parallel
run is inter-agent information flow, not an added primary signal (which parallel
Design G also withholds). Task-generic: no label names are hardcoded.
"""

from __future__ import annotations

import re
from typing import Any, List

# Order in which specialist slots are surfaced (matches orchestrator run order).
_CHAIN_SLOTS: List[tuple[str, str]] = [
    ("lexical_output", "Lexical agent"),
    ("logic_output", "Polarity agent"),
    ("contextual_output", "Contextual agent"),
]

_MAX_NOTE_CHARS = 160


def _sanitize(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= _MAX_NOTE_CHARS:
        return cleaned
    return cleaned[:_MAX_NOTE_CHARS].rstrip() + "..."


def build_agent_chain_block(
    state: Any, exclude_slot: str, analysis_kind: str, style: str = "independent"
) -> str:
    """Render prior specialist outputs as a context block, or ``""`` if none.

    Parameters
    ----------
    state:
        The ``PipelineState`` (read-only). Populated specialist slots preceding
        the current agent are surfaced.
    exclude_slot:
        The current agent's own output slot (e.g. ``"logic_output"``), so it never
        sees its own stale output.
    analysis_kind:
        The current agent's role word (``"lexical"`` / ``"polarity"`` /
        ``"contextual"``) — used only in the framing wording, never a label.
    style:
        ``"independent"`` (default) — anti-anchoring framing: use the prior
        conclusions as context only, do not copy. ``"collaborative"`` — invites
        the agent to actually build on / refine the prior conclusions. This is the
        knob that decides whether the chain discourages or encourages using the
        predecessors' outputs.
    """
    lines: List[str] = []
    for slot, display in _CHAIN_SLOTS:
        if slot == exclude_slot:
            continue
        out = getattr(state, slot, None)
        if out is None:
            continue
        mo = getattr(out, "model_output", None)
        if mo is None or mo.label is None or mo.confidence is None:
            continue
        note = _sanitize(getattr(out, "notes", "") or "")
        entry = f"  {display} -> label='{mo.label}', confidence={mo.confidence:.3f}"
        if note:
            # No wrapping quotes around the note: notes are free-text and often
            # contain apostrophes (e.g. "others' dislikes"), which would collide
            # with a quote delimiter. A colon separator is unambiguous.
            entry += f" — note: {note}"
        lines.append(entry)

    if not lines:
        return ""

    if style == "collaborative":
        header = (
            "PRIOR AGENTS' ANALYSES (input from other specialist agents on this "
            f"same text — take them into account for your {analysis_kind} analysis):"
        )
        footer = (
            "Treat these as conclusions from teammate specialists. Take them into "
            f"account: where they surface cues or context relevant to your "
            f"{analysis_kind} judgment, incorporate them; where you have a stronger "
            "reason from the text, refine or correct them. Aim for the best combined "
            "judgment of the author's intended label for this task, not an isolated one."
        )
    else:  # "independent" (default) — anti-anchoring
        header = (
            "PRIOR AGENTS' ANALYSES (context only — you are an independent "
            f"{analysis_kind} adjudicator running later in the chain):"
        )
        footer = (
            f"Do your own {analysis_kind} analysis of the text FIRST. These prior "
            "agents may be wrong; use their conclusions as context only, agree only "
            "when the text evidence supports them, and choose a different allowed "
            "label if the evidence points elsewhere. Do NOT simply copy them."
        )
    return "\n".join([header, *lines, footer])
