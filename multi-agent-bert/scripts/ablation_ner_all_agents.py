"""scripts/ablation_ner_all_agents.py

UNIFIED Exp-B ablation: ALL six agent roles evaluated on the fine-tuned primary,
real Sabty test, entity-level (seqeval). Fixed few-shot (no dynamic). Each row is
the primary transformed/augmented by one role (or a small combination):

  primary                — the fine-tuned model alone
  + reflection           — LLM reviews & corrects the primary's draft (rewrite)
  + debate               — LLM judges primary vs LLM-span where they disagree
  + disambiguation       — LLM re-types the primary's detected entities
  + gaz+verify           — gazetteer proposes, verify filters (add)
  + llm+verify           — LLM-span proposes (few-shot), verify filters (add)
  + gaz+llm+verify       — both proposers, verified (add)

    python scripts/ablation_ner_all_agents.py --model_dir models/xlmr_sabty_ner \\
        --limit 200 --env_file ../Modified_Version/.env
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.llm_ner_agent import coerce_to_valid
from src.agents.llm_ner_span_agent import (
    LLMNERSpanAgent, _norm, align_entities_to_tokens, build_span_prompt, spans_from_tags,
)
from src.agents.llm_ner_reflection_agent import LLMNERReflectionAgent
from src.agents.llm_ner_panel_agents import LLMNERDebateAgent, LLMNERDisambiguationAgent
from src.agents.ner_retrieval_agents import LLMNERVerifyAgent, build_gazetteer_from_conll
from src.evaluation.ner_conll_loader import TYPE_LABELS, load_conll, tag_to_type, to_type_dataset
from src.llm.openai_client import OpenAIClient
from src.models.transformer_ner_tagger import TransformerNERTagger
from src.pipeline.ner_agentic import agentic_ner_task_config
from src.state.schema import (AgentOutput, ModelOutput, PipelineState,
    SequenceLabelingOutput, StateMetadata, TokenTag)

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "NER" / "Train_AR-EN_NER.txt"
TEST = ROOT / "data" / "NER" / "Test_AR-EN_NER.txt"
DESC = {t: f"a named entity of type {t}" for t in TYPE_LABELS if t != "O"}


def to_bio(tags):
    out, prev = [], "O"
    for t in tags:
        out.append("O" if t == "O" else (f"I-{t}" if t == prev else f"B-{t}"))
        prev = t
    return out


def _load_env(env_file):
    for c in ([env_file] if env_file else [".env"]):
        p = Path(c) if c else None
        if p and not p.is_absolute():
            p = ROOT / c
        if p and p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return str(p)
    return None


def pick_examples(train, k=2, max_len=22):
    picked, seen = [], set()
    for want in [l for l in TYPE_LABELS if l != "O"]:
        cnt = 0
        for i, s in enumerate(train):
            if i in seen or len(s["tokens"]) > max_len:
                continue
            ents = spans_from_tags(s["tokens"], [tag_to_type(t) for t in s["tags"]])
            if ents and any(e["type"] == want for e in ents):
                picked.append((s["text"], ents)); seen.add(i); cnt += 1
                if cnt >= k:
                    break
    return picked


def gaz_full(tokens, gaz, valid):
    norm = [_norm(t) for t in tokens]
    tags = ["O"] * len(tokens)
    maxlen = max((len(k.split()) for k in gaz), default=1)
    i = 0
    while i < len(tokens):
        hit = False
        for L in range(min(maxlen, len(tokens) - i), 0, -1):
            key = " ".join(norm[i:i + L])
            if key in gaz:
                ty = coerce_to_valid(gaz[key], valid)
                if ty != "O":
                    for j in range(i, i + L):
                        tags[j] = ty
                    i += L; hit = True; break
        if not hit:
            i += 1
    return tags


def gap_spans(tokens, prim, cand):
    spans, i, n = [], 0, len(tokens)
    while i < n:
        if prim[i] == "O" and cand[i] != "O":
            ty, j = cand[i], i
            while j < n and prim[j] == "O" and cand[j] == ty:
                j += 1
            spans.append({"start": i, "end": j, "type": ty, "text": " ".join(tokens[i:j])})
            i = j
        else:
            i += 1
    return spans


def apply_spans(prim, spans):
    out = list(prim)
    for s in spans:
        for j in range(s["start"], s["end"]):
            out[j] = s["type"]
    return out


def _key(t):
    return " ".join(_norm(x) for x in t.split())


def ao(tokens, tags):
    return AgentOutput(agent_name="x", model_output=ModelOutput(),
        sequence_output=SequenceLabelingOutput(
            tags=[TokenTag(token=t, tag=g, confidence=0.9) for t, g in zip(tokens, tags)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", default="models/xlmr_sabty_ner")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--env_file", default=None)
    args = ap.parse_args()

    from seqeval.metrics import f1_score, precision_score, recall_score

    train = load_conll(TRAIN)
    gaz = build_gazetteer_from_conll(train, tag_to_type)
    examples = pick_examples(train)
    sents = to_type_dataset(load_conll(TEST))[:args.limit]
    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir, device=args.device, tag_normalizer=tag_to_type)
    if not _load_env(args.env_file) or not os.environ.get("OPENAI_API_KEY"):
        print("[!] no OPENAI_API_KEY — pass --env_file"); return
    client = OpenAIClient(model=args.llm_model, max_tokens=1500)
    valid = set(TYPE_LABELS)

    span = LLMNERSpanAgent(client, output_slot="contextual", descriptions=DESC, examples=examples)
    reflector = LLMNERReflectionAgent(client, output_slot="contextual")
    debate = LLMNERDebateAgent(client, source_a="model", source_b="contextual", output_slot="contextual")
    disamb = LLMNERDisambiguationAgent(client, source_slot="contextual", output_slot="contextual")

    def mk(tokens, text):
        return PipelineState(metadata=StateMetadata(sample_id="x"), input_text=text,
            task_config=agentic_ner_task_config(TYPE_LABELS, threshold=1.0, label_descriptions=DESC),
            extras={"tokens": tokens})

    cfgs = ["primary", "+reflection", "+debate", "+disambiguation",
            "+gaz+verify", "+llm+verify", "+gaz+llm+verify"]
    g = []
    pred = {c: [] for c in cfgs}
    for s in sents:
        toks, text = s["tokens"], s["text"]
        prim = [t.tag for t in tagger.tag(toks, task_labels=TYPE_LABELS).tags]
        g.append(to_bio(s["tags"]))

        # LLM span (fixed few-shot) proposals
        st = mk(toks, text); span.run(st)
        llm_full = [t.tag for t in st.contextual_output.sequence_output.tags]

        # reflection: correct the primary draft
        st = mk(toks, text); st.ner_model_output = ao(toks, prim); reflector.run(st)
        refl = [t.tag for t in st.contextual_output.sequence_output.tags]

        # debate: primary vs llm-span
        st = mk(toks, text); st.ner_model_output = ao(toks, prim); st.contextual_output = ao(toks, llm_full)
        debate.run(st)
        deb = [t.tag for t in st.contextual_output.sequence_output.tags]

        # disambiguation: re-type the primary's entities
        st = mk(toks, text); st.contextual_output = ao(toks, prim); disamb.run(st)
        dis = [t.tag for t in st.contextual_output.sequence_output.tags]

        # additions + verify
        gaz_adds = gap_spans(toks, prim, gaz_full(toks, gaz, valid))
        llm_adds = gap_spans(toks, prim, align_entities_to_tokens(toks, [], valid)) if False else \
                   gap_spans(toks, prim, llm_full)
        union = gaz_adds + [a for a in llm_adds
                            if not any(a["start"] == d["start"] and a["end"] == d["end"] for d in gaz_adds)]
        conf = set()
        if union:
            vraw = client.generate(LLMNERVerifyAgent._prompt(text, union))
            try:
                conf = {_key(e["text"]) for e in LLMNERVerifyAgent._parse_kept(vraw, union)}
            except Exception:
                conf = {_key(a["text"]) for a in union}
        v = lambda spans: [a for a in spans if _key(a["text"]) in conf]

        pred["primary"].append(to_bio(prim))
        pred["+reflection"].append(to_bio(refl))
        pred["+debate"].append(to_bio(deb))
        pred["+disambiguation"].append(to_bio(dis))
        pred["+gaz+verify"].append(to_bio(apply_spans(prim, v(gaz_adds))))
        pred["+llm+verify"].append(to_bio(apply_spans(prim, v(llm_adds))))
        pred["+gaz+llm+verify"].append(to_bio(apply_spans(prim, v(gaz_adds + llm_adds))))

    print(f"\nSentences: {len(sents)}  LLM calls: {client.usage_summary()['calls']}"
          f"  cost≈${client.usage_summary()['est_cost_usd']:.4f}")
    print("\n" + "=" * 58)
    print(f"{'agent panel':<22}{'F1':>8}{'precision':>11}{'recall':>9}")
    print("-" * 58)
    base = f1_score(g, pred["primary"])
    for c in cfgs:
        f1 = f1_score(g, pred[c])
        d = "" if c == "primary" else f"  ({f1-base:+.3f})"
        print(f"{c:<22}{f1:>8.3f}{precision_score(g, pred[c]):>11.3f}{recall_score(g, pred[c]):>9.3f}{d}")
    print("=" * 58)


if __name__ == "__main__":
    main()
