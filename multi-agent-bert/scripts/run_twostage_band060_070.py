"""Two-stage Topic-540 (gen540@136 -> silver_hybrid_train, seed 42) + full
agentic pipeline at tau=0.60, on its OWN established test set:
silver_hybrid_test (300 rows). Test set NOT changed.

Checkpoint rebuilt and verified bit-identical to the recorded original
(acc 0.7767 / macroF1 0.7596, 300/300 predictions and confidences match).
tau=0.60 routes 50 rows (16.7%). NOTE: n=50 routed is a thin sample; treat the
net-gain estimate as directional only.

Calls evaluate_pipeline.main() unchanged; observation-only audit patches.
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

OUT_DIR = ROOT / "experiments/outputs/multi_agent_bert/experiment_twostage_agentic_tau070_band"
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
    "--dataset", "experiments/outputs/multi_agent_bert/experiment_twostage_agentic_tau070_band/band_060_070.jsonl",
    "--config", "src/config/default.yaml",
    "--active_task", "topic_classification",
    "--pipeline_mode", "full_agentic",
    "--mode", "both",
    "--threshold", "1.01",
    "--primary_model", "transformer",
    "--transformer_checkpoint", "D:/switchlingua_checkpoints/twostage_gs_seed42",
    "--transformer_device", "cuda",
    "--llm_client", "openai",
    "--llm_model", "gpt-4o-mini",
    "--output_dir", str(OUT_DIR / "pipeline_out"),
    "--run_id", "twostage_band060_070_seed42",
]

print("[audit] argv:", " ".join(ARGV), flush=True)
rc = ep.main(ARGV)
_calls_fh.close()
_rows_fh.close()
print(f"[audit] evaluate_pipeline.main returned {rc}")
print(f"[audit] LLM calls recorded : {_call_seq['n']} -> {AUDIT_CALLS}")
print(f"[audit] row states recorded -> {AUDIT_ROWS}")
sys.exit(rc)
