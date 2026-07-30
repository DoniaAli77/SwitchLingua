"""scripts/evaluate_ner_llm.py

Full-agentic NER evaluation: the sentiment/topic-style pipeline applied to NER.

    XLM-R primary  ->  router (min-confidence)  ->  escalate  ->  LLM NER agent
                                                              ->  consensus (XLM-R + LLM)

Compares:
  1. XLMR_PRIMARY   — XLM-R alone (primary_only). Baseline.
  2. FUSED_HEUR     — XLM-R + heuristic agents (routed). The rules degrade Arabic.
  3. FUSED_LLM      — XLM-R + LLM NER agent (routed, heuristics OFF). The LLM
                      reads Arabic, so escalation should help instead of hurt.

Only sentences the router marks UNCERTAIN call the LLM, so cost is tiny
(~1 gpt-4o-mini call per escalated sentence).

API key
-------
Needs OPENAI_API_KEY (env or a .env file via --env_file). On a corporate proxy
you may also need OPENAI_BASE_URL / HTTPS_PROXY / REQUESTS_CA_BUNDLE. If no key
is found the LLM config is skipped and only the XLM-R baseline runs.

Usage
-----
    python scripts/evaluate_ner_llm.py
    python scripts/evaluate_ner_llm.py --threshold 0.95 --model_dir C:/models/xlmr-ner-hrl
    python scripts/evaluate_ner_llm.py --env_file ../Modified_Version/.env
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.consensus_agent import ConsensusAgent
from src.agents.contextual_agent import ContextualAgent
from src.agents.explainability_agent import ExplainabilityAgent
from src.agents.lexical_agent import LexicalAgent
from src.agents.logic_agent import LogicAgent
from src.agents.llm_ner_agent import LLMNERAgent
from src.agents.llm_ner_reflection_agent import LLMNERReflectionAgent
from src.agents.ner_consensus_agent import NERConsensusAgent
from src.agents.ner_contextual_agent import NERContextualAgent
from src.agents.ner_lexical_agent import NERLexicalAgent
from src.agents.ner_logic_agent import NERLogicAgent
from src.evaluation.ner_evaluator import NEREvaluator
from src.llm.mock_client import MockLLMClient
from src.llm.openai_client import OpenAIClient
from src.models.transformer_ner_tagger import TransformerNERTagger
from src.pipeline.ner_agentic import build_agentic_ner_orchestrator
from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.router import Router
from src.state.schema import ModelOutput, TaskConfig

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
DATA = Path(__file__).resolve().parent.parent / "data" / "dev_dummy_ner.jsonl"
DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "experiments" / "outputs" / "multi_agent_bert" / "experiment_ner_llm"
)

_GAZ = {"ORG": ["Google", "Microsoft", "Amazon", "Apple", "OpenAI", "Tesla", "UNESCO", "WHO", "FIFA"],
        "LOC": ["Paris", "Cairo", "Dubai", "London", "Berlin", "Geneva", "Riyadh", "Qatar", "Egypt"],
        "PER": ["Ahmed", "Sara", "Mohamed", "Omar", "Fatima", "Ali", "Elon", "Musk"]}
_RULES = {"PER": [r"(Dr|Mr|Ms|Mrs|CEO)\.?"]}
_KNOWN = {"google": "ORG", "cairo": "LOC", "london": "LOC"}


class _StubPrimary:
    def run(self, state):
        state.primary_model_output = ModelOutput(label="O", confidence=0.9,
                                                 probabilities={"O": 1.0})
        return state


def _load_env(env_file: str | None):
    """Populate OPENAI_* from a .env file (best-effort, no dependency)."""
    candidates = [env_file] if env_file else [".env", "../.env"]
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent / c
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return str(p)
    return None


def _base():
    llm = MockLLMClient(mode="label_echo", allowed_labels=LABELS)
    return dict(
        primary_classifier=_StubPrimary(), router=Router(),
        lexical_agent=LexicalAgent(), contextual_agent=ContextualAgent(llm_client=llm),
        logic_agent=LogicAgent(), consensus_agent=ConsensusAgent(),
        explainability_agent=ExplainabilityAgent(),
    )


def _primary_orch(tagger):
    return PipelineOrchestrator(**_base(), ner_primary=tagger)


def _fused_heur_orch(tagger, model_weight):
    return PipelineOrchestrator(
        **_base(),
        ner_lexical_agent=NERLexicalAgent(gazetteer=_GAZ),
        ner_logic_agent=NERLogicAgent(rule_map=_RULES),
        ner_contextual_agent=NERContextualAgent(known_entities=_KNOWN),
        ner_consensus_agent=NERConsensusAgent(weights={"model": model_weight}),
        ner_primary=tagger)


def _fused_llm_orch(tagger, llm_client, model_weight, llm_weight):
    # Canonical agentic lineup with the "tagger" specialist (votes with the model).
    return build_agentic_ner_orchestrator(
        tagger, llm_client, specialist="tagger",
        model_weight=model_weight, llm_weight=llm_weight)


def _fused_reflect_orch(tagger, llm_client):
    # Canonical agentic lineup with the "reflector" specialist; weight only the
    # reflection output so the corrected draft is the final answer on escalation.
    return build_agentic_ner_orchestrator(
        tagger, llm_client, specialist="reflector",
        model_weight=0.0, llm_weight=1.0)


def _run(name, orch, data, mode, threshold, out_dir):
    tc = TaskConfig(task_name="ner", task_type="sequence_labeling",
                    labels=LABELS, pipeline_mode=mode, threshold=threshold)
    ev = NEREvaluator(task_config=tc, orchestrator=orch, run_id=name)
    rep = ev.evaluate(data)
    ev.save(rep, output_dir=str(out_dir))
    print(f"\n=== {name} ({mode}) ===")
    print(f"  token_accuracy={rep.token_accuracy:.3f}  macro_f1={rep.macro_f1:.3f}"
          f"  errors={rep.meta['error_samples']}")
    for m in rep.per_tag:
        print(f"    {m.label:6s} P={m.precision:.2f} R={m.recall:.2f} F1={m.f1:.2f}")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Davlan/xlm-roberta-base-ner-hrl")
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--model_weight", type=float, default=1.0)
    ap.add_argument("--llm_weight", type=float, default=1.0)
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--env_file", default=None)
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    data = [json.loads(l) for l in DATA.open(encoding="utf-8")]
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading XLM-R NER model: {args.model_dir or args.model} ...")
    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir or args.model, device=args.device)

    reps = {}
    reps["xlmr_primary"] = _run("ner_xlmr_primary", _primary_orch(tagger), data,
                                "primary_only", args.threshold, out_dir)
    reps["fused_heur"] = _run("ner_fused_heur", _fused_heur_orch(tagger, args.model_weight),
                              data, "paper_style", args.threshold, out_dir)

    env_used = _load_env(args.env_file)
    if os.environ.get("OPENAI_API_KEY"):
        if env_used:
            print(f"\n[env loaded from {env_used}]")
        llm_client = OpenAIClient(model=args.llm_model)
        orch = _fused_llm_orch(tagger, llm_client, args.model_weight, args.llm_weight)
        reps["fused_llm_vote"] = _run("ner_fused_llm_vote", orch, data,
                                      "full_agentic", args.threshold, out_dir)

        # Reflection: LLM reviews & corrects the primary draft (review-and-correct).
        reflect_client = OpenAIClient(model=args.llm_model)
        orch_r = _fused_reflect_orch(tagger, reflect_client)
        reps["fused_reflect"] = _run("ner_fused_reflect", orch_r, data,
                                     "full_agentic", args.threshold, out_dir)
        try:
            print("\nLLM usage (vote):", llm_client.usage_summary())
            print("LLM usage (reflect):", reflect_client.usage_summary())
        except Exception:
            pass
    else:
        print("\n[!] OPENAI_API_KEY not found — skipping the FUSED_LLM run.")
        print("    Set it in the environment or pass --env_file <path/to/.env>.")

    print("\n" + "=" * 52)
    print(f"{'CONFIG':<18}{'token_acc':>12}{'macro_f1':>12}")
    print("-" * 52)
    for key, rep in reps.items():
        print(f"{key:<18}{rep.token_accuracy:>12.3f}{rep.macro_f1:>12.3f}")
    print("=" * 52)
    print(f"\nOutputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
