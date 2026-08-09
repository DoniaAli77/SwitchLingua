"""MAIN SwitchLingua experiment:
Topic-540 XLM-R primary + real full_agentic pipeline at the primary-calibrated
threshold tau = 0.30.

tau=0.30 was chosen LABEL-BLIND, from the saved primary confidence column only,
to give selective coverage (169/1163 = 14.53%) consistent with the operating
range already established for the sentiment primaries in EXPERIMENT_REGISTRY.md
("threshold calibrated per primary": XLM-R 0.9, Ahmed 0.7, C3 0.9 -> 5-23%
escalation). It was NOT optimised against Silver correctness or any agentic
outcome, and no performance-based sweep was run.

All 1,163 rows are fed to the pipeline and the FROZEN Router applies tau itself
(route iff primary confidence < 0.30). Rows are NOT pre-selected by this
script. Rows at or above 0.30 take the accept_primary fast path, keep the
primary prediction, and make zero LLM calls.

This script does NOT reimplement the pipeline. It imports `evaluate_pipeline`
and calls its `main()` unchanged, so the frozen orchestrator/prompts/consensus/
weights/preprocessing are used exactly as-is. Three OBSERVATION-ONLY monkey
patches are installed to capture the audit trail that evaluate_pipeline.py does
not persist:

  1. OpenAIClient.generate  -> records (sample_id, agent, prompt, raw_response,
                               timestamps, latency, error) for every LLM call.
  2. <each LLM agent>.run   -> tags the currently-executing agent (label only).
  3. PipelineOrchestrator.run -> tags the current sample id and, after the run,
                               dumps the full PipelineState (primary output +
                               each agent's parsed label/confidence + consensus
                               votes/rationale + final output + routing).

None of the patches alter arguments, return values, control flow, or prompts.
Each wrapper calls the original function and returns its result untouched.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT_DIR = ROOT / "experiments/outputs/multi_agent_bert/experiment_gen540_agentic_silver1163_tau030"
OUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_CALLS = OUT_DIR / "audit_llm_calls.jsonl"       # one record per LLM call
AUDIT_ROWS = OUT_DIR / "audit_rows.jsonl"             # one record per sample

import evaluate_pipeline as ep
from src.llm.openai_client import OpenAIClient
from src.pipeline.orchestrator import PipelineOrchestrator
from src.agents.llm_lexical_agent import LLMLexicalAgent
from src.agents.llm_logic_agent import LLMLogicAgent
from src.agents.contextual_agent import ContextualAgent
from src.agents.llm_explainability_agent import LLMExplainabilityAgent

# ---------------------------------------------------------------- audit state
_CURRENT = {"sample_id": None, "agent": None}
_call_seq = {"n": 0}
_calls_fh = open(AUDIT_CALLS, "w", encoding="utf-8")
_rows_fh = open(AUDIT_ROWS, "w", encoding="utf-8")


# --- patch 1: record every LLM call (prompt in, raw response out) ------------
_orig_generate = OpenAIClient.generate


def _audited_generate(self, prompt: str) -> str:
    _call_seq["n"] += 1
    idx = _call_seq["n"]
    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat()
    err = None
    response = ""
    try:
        response = _orig_generate(self, prompt)          # unchanged call
        return response
    except Exception as exc:                              # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _calls_fh.write(json.dumps({
            "call_index": idx,
            "sample_id": _CURRENT["sample_id"],
            "agent": _CURRENT["agent"],
            "model": self.model,
            "started_utc": started,
            "latency_sec": round(time.time() - t0, 3),
            "prompt": prompt,                             # FULL prompt as sent
            "raw_response": response,                     # FULL raw response
            "error": err,
        }, ensure_ascii=False) + "\n")
        _calls_fh.flush()


OpenAIClient.generate = _audited_generate


# --- patch 2: tag which agent is executing (label only, no behaviour change) --
def _tag_agent(cls, name):
    orig = cls.run

    def wrapper(self, state):
        _CURRENT["agent"] = name
        return orig(self, state)                          # unchanged call

    wrapper.__wrapped__ = orig
    cls.run = wrapper


for _cls, _name in [
    (LLMLexicalAgent, "lexical_agent"),
    (LLMLogicAgent, "logic_agent"),
    (ContextualAgent, "contextual_agent"),
    (LLMExplainabilityAgent, "explainability_agent"),
]:
    _tag_agent(_cls, _name)


# --- patch 3: tag sample id + dump full state after each pipeline run --------
_orig_run = PipelineOrchestrator.run


def _mo(m):
    if m is None:
        return None
    return {"label": m.label, "confidence": m.confidence,
            "probabilities": m.probabilities, "raw_text": m.raw_text}


def _ao(a):
    if a is None:
        return None
    return {"agent_name": a.agent_name, "parsed": _mo(a.model_output),
            "notes": a.notes, "features": a.features}


def _audited_run(self, state):
    _CURRENT["sample_id"] = getattr(state.metadata, "sample_id", None)
    _CURRENT["agent"] = "primary/router"
    out = _orig_run(self, state)                          # unchanged call
    try:
        r = out.routing_info
        c = out.consensus_output
        _rows_fh.write(json.dumps({
            "sample_id": getattr(out.metadata, "sample_id", None),
            "input_text": out.input_text,
            "primary": _mo(out.primary_model_output),
            "routing": None if r is None else {
                "threshold": r.threshold, "decision": r.decision,
                "reason": getattr(r, "reason", None)},
            "lexical_output": _ao(out.lexical_output),
            "logic_output": _ao(out.logic_output),
            "contextual_output": _ao(out.contextual_output),
            "consensus": None if c is None else {
                "label": c.label, "confidence": c.confidence,
                "votes": c.votes, "rationale": c.rationale},
            "explanation": None if out.explanation_output is None else {
                "text": getattr(out.explanation_output, "explanation", None)},
            "final": None if out.final_output is None else {
                "label": out.final_output.label,
                "confidence": out.final_output.confidence},
            "history_stages": [getattr(h, "component", getattr(h, "stage", None))
                                for h in out.history],
        }, ensure_ascii=False) + "\n")
        _rows_fh.flush()
    except Exception as exc:                              # noqa: BLE001
        print(f"[audit] state dump failed for {_CURRENT['sample_id']}: {exc}",
              file=sys.stderr)
    return out


PipelineOrchestrator.run = _audited_run

# ---------------------------------------------------------------- run it
ARGV = [
    "--dataset", "experiments/outputs/multi_agent_bert/experiment_silver_topic540/silver_full1163_ordered.jsonl",
    "--config", "src/config/default.yaml",
    "--active_task", "topic_classification",
    "--pipeline_mode", "full_agentic",
    "--mode", "both",
    "--threshold", "0.30",
    "--primary_model", "transformer",
    "--transformer_checkpoint", "experiments/checkpoints/topic_gen540_xlmr",
    "--transformer_device", "cuda",
    "--llm_client", "openai",
    "--llm_model", "gpt-4o-mini",
    "--output_dir", str(OUT_DIR / "pipeline_out"),
    "--run_id", "gen540_agentic_silver1163_tau030",
]

print("[audit] argv:", " ".join(ARGV), flush=True)
rc = ep.main(ARGV)
_calls_fh.close()
_rows_fh.close()
print(f"[audit] evaluate_pipeline.main returned {rc}")
print(f"[audit] LLM calls recorded : {_call_seq['n']} -> {AUDIT_CALLS}")
print(f"[audit] row states recorded -> {AUDIT_ROWS}")
sys.exit(rc)
