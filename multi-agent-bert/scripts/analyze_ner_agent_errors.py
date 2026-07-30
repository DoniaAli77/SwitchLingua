"""scripts/analyze_ner_agent_errors.py

Diagnose WHERE the agent panel loses accuracy vs the fine-tuned primary.

For each escalated sentence it runs primary -> reflector -> debate ->
disambiguation and, for every token the agents CHANGED (primary_tag != final_tag),
categorises the change against gold:

  HELPED  : primary wrong -> agent right
  HURT    : primary right -> agent wrong
  NEUTRAL : both wrong (just different)

and, for HURT, the failure MODE:
  deleted   : entity -> O            (agent removed a correct entity)
  added     : O -> entity            (agent invented an entity)
  retyped   : typeA -> typeB         (agent changed a correct type)

Then prints concrete HURT examples (with sentence context) so we can tell whether
the cause is bad prompts, annotation-convention mismatch, or over-eager overrides.

Usage
-----
    python scripts/analyze_ner_agent_errors.py --limit 60 --env_file ../Modified_Version/.env
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.llm_ner_panel_agents import LLMNERDebateAgent, LLMNERDisambiguationAgent
from src.agents.llm_ner_reflection_agent import LLMNERReflectionAgent
from src.evaluation.ner_conll_loader import TYPE_LABELS, load_conll, tag_to_type, to_type_dataset
from src.llm.openai_client import OpenAIClient
from src.models.transformer_ner_tagger import TransformerNERTagger
from src.pipeline.ner_agentic import agentic_ner_task_config
from src.state.schema import PipelineState, StateMetadata

TEST = Path(__file__).resolve().parent.parent / "data" / "NER" / "Test_AR-EN_NER.txt"
_DESC = {
    "PERS": "a person's name (Arabic or English)",
    "LOC": "a location/city/country/place", "ORG": "an organization/company/team",
    "MISC": "a named entity that is NOT person/location/organization (events, "
            "nationalities, products, competitions, titles)",
}


def _load_env(env_file):
    for c in ([env_file] if env_file else [".env", "../.env"]):
        p = Path(c) if c else None
        if p and not p.is_absolute():
            p = Path(__file__).resolve().parent.parent / c
        if p and p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return str(p)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", default="models/xlmr_sabty_ner")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--env_file", default=None)
    ap.add_argument("--show", type=int, default=18, help="how many HURT examples to print")
    args = ap.parse_args()

    sents = to_type_dataset(load_conll(TEST))[:args.limit]
    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir, device=args.device, tag_normalizer=tag_to_type)
    if not _load_env(args.env_file) or not os.environ.get("OPENAI_API_KEY"):
        print("[!] no OPENAI_API_KEY — pass --env_file"); return
    client = OpenAIClient(model=args.llm_model, max_tokens=1500)
    reflector = LLMNERReflectionAgent(client, output_slot="contextual")
    debate = LLMNERDebateAgent(client, source_a="model", source_b="contextual", output_slot="contextual")
    disambig = LLMNERDisambiguationAgent(client, source_slot="contextual", output_slot="contextual")

    verdict = Counter()          # helped / hurt / neutral
    hurt_mode = Counter()        # deleted / added / retyped
    help_mode = Counter()
    hurt_examples = []

    for s in sents:
        toks, gold = s["tokens"], s["tags"]
        st = PipelineState(metadata=StateMetadata(sample_id=s["id"]), input_text=s["text"],
                           task_config=agentic_ner_task_config(TYPE_LABELS, threshold=args.threshold,
                                                               label_descriptions=_DESC),
                           extras={"tokens": toks})
        tagger.run(st)
        prim = [t.tag for t in st.ner_model_output.sequence_output.tags]
        confs = [t.confidence for t in st.ner_model_output.sequence_output.tags]
        if confs and min(confs) >= args.threshold:
            continue
        reflector.run(st); debate.run(st); disambig.run(st)
        final = [t.tag for t in st.contextual_output.sequence_output.tags]

        for i, (p, f, g) in enumerate(zip(prim, final, gold)):
            if p == f:
                continue
            mode = ("deleted" if f == "O" else "added" if p == "O" else "retyped")
            if p != g and f == g:
                verdict["helped"] += 1; help_mode[mode] += 1
            elif p == g and f != g:
                verdict["hurt"] += 1; hurt_mode[mode] += 1
                if len(hurt_examples) < args.show:
                    ctx = " ".join(toks[max(0, i-3):i+4])
                    hurt_examples.append((toks[i], g, p, f, mode, ctx))
            else:
                verdict["neutral"] += 1

    print(f"\nAnalyzed escalated sentences from {len(sents)} loaded.  "
          f"LLM calls: {client.usage_summary()['calls']}  cost≈${client.usage_summary()['est_cost_usd']:.4f}")
    print("\n=== Changes the agents made (primary != final) ===")
    tot = sum(verdict.values())
    for k in ("helped", "hurt", "neutral"):
        v = verdict[k]
        print(f"  {k:<8} {v:>4}  ({100*v/tot:.0f}%)" if tot else f"  {k}: 0")
    print(f"  net (helped - hurt) = {verdict['helped'] - verdict['hurt']:+d}")
    print("\n=== HURT breakdown (primary was RIGHT, agent broke it) ===")
    for m in ("deleted", "added", "retyped"):
        print(f"  {m:<8} {hurt_mode[m]}")
    print("\n=== HELP breakdown (agent fixed a primary error) ===")
    for m in ("deleted", "added", "retyped"):
        print(f"  {m:<8} {help_mode[m]}")

    print(f"\n=== {len(hurt_examples)} concrete HURT cases (token: gold | primary->final [mode]) ===")
    for tok, g, p, f, mode, ctx in hurt_examples:
        print(f"  '{tok}'  gold={g}  {p}->{f} [{mode}]   ...{ctx}...")


if __name__ == "__main__":
    main()
