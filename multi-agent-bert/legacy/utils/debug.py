"""utils/debug.py

Debugging helpers for the multi-agent classification pipeline.
"""

from __future__ import annotations

from src.state.schema import PipelineState

_SEP = "─" * 56


def print_debug_summary(state: PipelineState) -> None:
    """Pretty-print key pipeline state information for a single sample.

    Parameters
    ----------
    state:
        A ``PipelineState`` after ``PipelineOrchestrator.run()`` has returned.
        The state is never mutated.
    """
    primary = state.primary_model_output
    routing = state.routing_info
    final = state.final_output
    escalated = routing is not None and routing.decision == "escalate"

    # Agents that actually wrote an output (non-None fields).
    _agent_fields = {
        "lexical":      state.lexical_output,
        "logic":        state.logic_output,
        "contextual":   state.contextual_output,
        "deliberation": state.deliberation_output,
        "consensus":    state.consensus_output,
    }
    agents_ran = [name for name, val in _agent_fields.items() if val is not None]

    # Fall back to history component names when output fields are empty
    # (e.g. primary_only mode which skips specialists entirely).
    if not agents_ran:
        agents_ran = list(dict.fromkeys(
            e.component for e in state.history
            if e.component not in ("router", "explainability_agent", "primary_classifier")
        ))

    print(_SEP)
    print(f"  INPUT       : {state.input_text[:80]}")
    print(_SEP)
    print(f"  PRIMARY     : {primary.label or 'N/A'}"
          f"  (conf={primary.confidence:.3f})" if primary.confidence is not None
          else f"  PRIMARY     : {primary.label or 'N/A'}")
    print(f"  ROUTING     : {routing.decision if routing else 'N/A'}"
          f"  (threshold={routing.threshold:.2f})" if routing else
          f"  ROUTING     : N/A")
    print(f"  ESCALATED   : {'yes' if escalated else 'no'}")
    print(f"  AGENTS RAN  : {', '.join(agents_ran) if agents_ran else 'none'}")
    print(f"  FINAL LABEL : {final.label if final else 'N/A'}"
          f"  (conf={final.confidence:.3f})" if final and final.confidence is not None
          else f"  FINAL LABEL : {final.label if final else 'N/A'}")
    print(_SEP)
