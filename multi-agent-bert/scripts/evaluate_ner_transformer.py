"""scripts/evaluate_ner_transformer.py

End-to-end NER evaluation on the synthetic Arabic-English code-switched set
(data/dev_dummy_ner.jsonl), comparing three configurations of the NER path:

  1. HEURISTICS   — the hand-authored gazetteer/regex/known-entity agents only
                    (no real model), paper_style. Baseline from try_synthetic_ner.py.
  2. XLMR_PRIMARY — the real XLM-R NER model alone (primary_only mode).
  3. FUSED_ROUTED — the full symmetric pipeline: XLM-R primary → router
                    (min-confidence gate) → escalate to heuristic agents →
                    consensus (model heavily weighted). Mirrors sentiment/topic.

The model is ``Davlan/xlm-roberta-base-ner-hrl`` (XLM-R backbone, Arabic+English,
labels O/B-PER/I-PER/B-ORG/I-ORG/B-LOC/I-LOC — matches the synthetic set).

Offline / corporate-proxy notes
-------------------------------
The first run downloads ~1.1 GB from the HuggingFace Hub. If your SSL proxy
blocks it:
  * download the model once elsewhere and pass ``--model_dir C:/path/to/model``;
  * or set ``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1`` after it is cached;
  * or configure ``HTTPS_PROXY`` / ``REQUESTS_CA_BUNDLE`` for the proxy.

Usage
-----
    python scripts/evaluate_ner_transformer.py
    python scripts/evaluate_ner_transformer.py --model_dir C:/models/xlmr-ner-hrl
    python scripts/evaluate_ner_transformer.py --device cuda --threshold 0.6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.consensus_agent import ConsensusAgent
from src.agents.contextual_agent import ContextualAgent
from src.agents.explainability_agent import ExplainabilityAgent
from src.agents.lexical_agent import LexicalAgent
from src.agents.logic_agent import LogicAgent
from src.agents.ner_consensus_agent import NERConsensusAgent
from src.agents.ner_contextual_agent import NERContextualAgent
from src.agents.ner_lexical_agent import NERLexicalAgent
from src.agents.ner_logic_agent import NERLogicAgent
from src.evaluation.ner_evaluator import NEREvaluator
from src.llm.mock_client import MockLLMClient
from src.models.transformer_ner_tagger import TransformerNERTagger
from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.router import Router
from src.state.schema import ModelOutput, TaskConfig

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
DATA = Path(__file__).resolve().parent.parent / "data" / "dev_dummy_ner.jsonl"
DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "experiments" / "outputs" / "multi_agent_bert" / "experiment_ner_xlmr"
)

# Hand-authored, GENERAL-KNOWLEDGE heuristic resources (not copied from gold).
_GAZETTEER = {
    "ORG": ["Google", "Microsoft", "Amazon", "Apple", "OpenAI", "Tesla",
            "UNESCO", "WHO", "FIFA", "NASA"],
    "LOC": ["Paris", "Cairo", "Dubai", "London", "Berlin", "Geneva",
            "Riyadh", "Qatar", "Egypt", "Sudan", "Casablanca", "Mars"],
    "PER": ["Ahmed", "Sara", "Mohamed", "Omar", "Fatima", "Ali", "Elon", "Musk"],
}
_RULES = {"PER": [r"(Dr|Mr|Ms|Mrs|CEO)\.?"]}
_KNOWN = {
    "google": "ORG", "microsoft": "ORG", "amazon": "ORG", "apple": "ORG",
    "openai": "ORG", "tesla": "ORG", "unesco": "ORG", "who": "ORG",
    "paris": "LOC", "cairo": "LOC", "dubai": "LOC", "london": "LOC",
    "berlin": "LOC", "geneva": "LOC", "qatar": "LOC",
}


class _StubPrimary:
    """Placeholder classification primary (NER path ignores it)."""
    def run(self, state):
        state.primary_model_output = ModelOutput(
            label="O", confidence=0.9, probabilities={"O": 1.0})
        return state


def _base_kwargs():
    llm = MockLLMClient(mode="label_echo", allowed_labels=LABELS)
    return dict(
        primary_classifier=_StubPrimary(), router=Router(),
        lexical_agent=LexicalAgent(), contextual_agent=ContextualAgent(llm_client=llm),
        logic_agent=LogicAgent(), consensus_agent=ConsensusAgent(),
        explainability_agent=ExplainabilityAgent(),
    )


def _heuristic_orch():
    return PipelineOrchestrator(
        **_base_kwargs(),
        ner_lexical_agent=NERLexicalAgent(gazetteer=_GAZETTEER),
        ner_logic_agent=NERLogicAgent(rule_map=_RULES),
        ner_contextual_agent=NERContextualAgent(known_entities=_KNOWN),
        ner_consensus_agent=NERConsensusAgent(),
    )


def _primary_only_orch(tagger):
    return PipelineOrchestrator(**_base_kwargs(), ner_primary=tagger)


def _fused_orch(tagger, model_weight):
    return PipelineOrchestrator(
        **_base_kwargs(),
        ner_lexical_agent=NERLexicalAgent(gazetteer=_GAZETTEER),
        ner_logic_agent=NERLogicAgent(rule_map=_RULES),
        ner_contextual_agent=NERContextualAgent(known_entities=_KNOWN),
        ner_consensus_agent=NERConsensusAgent(weights={"model": model_weight}),
        ner_primary=tagger,
    )


def _tc(mode, threshold):
    return TaskConfig(task_name="ner", task_type="sequence_labeling",
                      labels=LABELS, pipeline_mode=mode, threshold=threshold)


def _run(name, orch, data, mode, threshold, out_dir):
    ev = NEREvaluator(task_config=_tc(mode, threshold), orchestrator=orch, run_id=name)
    rep = ev.evaluate(data)
    ev.save(rep, output_dir=str(out_dir))
    print(f"\n=== {name} ({mode}) ===")
    print(f"  samples={rep.num_samples} tokens={rep.num_tokens} errors={rep.meta['error_samples']}")
    print(f"  token_accuracy = {rep.token_accuracy:.3f}")
    print(f"  macro_f1       = {rep.macro_f1:.3f}")
    for m in rep.per_tag:
        print(f"    {m.label:6s} P={m.precision:.2f} R={m.recall:.2f} F1={m.f1:.2f} (support={m.support})")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Davlan/xlm-roberta-base-ner-hrl",
                    help="HuggingFace model id for the NER primary.")
    ap.add_argument("--model_dir", default=None,
                    help="Local model directory (overrides --model; offline).")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Router min-confidence gate for the fused run.")
    ap.add_argument("--model_weight", type=float, default=3.0,
                    help="Consensus weight for the model slot in the fused run.")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    data = [json.loads(l) for l in DATA.open(encoding="utf-8")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.model_dir or args.model
    print(f"Loading NER primary model: {checkpoint} (device={args.device}) ...")
    tagger = TransformerNERTagger.from_pretrained(checkpoint=checkpoint, device=args.device)

    reps = {}
    reps["heuristics"] = _run("ner_heuristics", _heuristic_orch(), data,
                              "paper_style", args.threshold, out_dir)
    reps["xlmr_primary"] = _run("ner_xlmr_primary", _primary_only_orch(tagger), data,
                                "primary_only", args.threshold, out_dir)
    reps["fused_routed"] = _run("ner_fused_routed",
                                _fused_orch(tagger, args.model_weight), data,
                                "paper_style", args.threshold, out_dir)

    print("\n" + "=" * 52)
    print(f"{'CONFIG':<18}{'token_acc':>12}{'macro_f1':>12}")
    print("-" * 52)
    for key, rep in reps.items():
        print(f"{key:<18}{rep.token_accuracy:>12.3f}{rep.macro_f1:>12.3f}")
    print("=" * 52)
    print(f"\nOutputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
