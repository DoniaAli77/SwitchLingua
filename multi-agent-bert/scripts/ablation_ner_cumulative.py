"""scripts/ablation_ner_cumulative.py

CUMULATIVE agent ablation: each step stacks on the previous (not independent).
Order: primary -> +gaz+verify -> +LLM+verify -> +debate -> +disambiguation -> +reflection.
Every step operates on the RUNNING result of the step before it. Real Sabty test,
entity-level (seqeval), fixed few-shot.

    python scripts/ablation_ner_cumulative.py --model_dir models/xlmr_sabty_ner --limit 200 \\
        --env_file ../Modified_Version/.env
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


def gap_spans(tokens, cur, cand):
    spans, i, n = [], 0, len(tokens)
    while i < n:
        if cur[i] == "O" and cand[i] != "O":
            ty, j = cand[i], i
            while j < n and cur[j] == "O" and cand[j] == ty:
                j += 1
            spans.append({"start": i, "end": j, "type": ty, "text": " ".join(tokens[i:j])})
            i = j
        else:
            i += 1
    return spans


def apply_spans(cur, spans):
    out = list(cur)
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
    ap.add_argument("--order", default="gaz,llm,debate,disambig,reflect",
                    help="comma-separated cumulative steps from: gaz,llm,debate,disambig,reflect")
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

    def verify(text, spans):
        if not spans:
            return set()
        vraw = client.generate(LLMNERVerifyAgent._prompt(text, spans))
        try:
            return {_key(e["text"]) for e in LLMNERVerifyAgent._parse_kept(vraw, spans)}
        except Exception:
            return {_key(a["text"]) for a in spans}

    STEP_LABEL = {"gaz": "gaz+verify", "llm": "LLM+verify", "debate": "debate",
                  "disambig": "disambig", "reflect": "reflection"}
    order = [s.strip() for s in args.order.split(",") if s.strip()]
    for st in order:
        if st not in STEP_LABEL:
            print(f"unknown step '{st}' — use any of {list(STEP_LABEL)}"); return
    cols, lbl = ["primary"], "primary"
    for st in order:
        lbl = f"{lbl} +{STEP_LABEL[st]}"
        cols.append(lbl)

    g = []
    pred = {c: [] for c in cols}
    for s in sents:
        toks, text = s["tokens"], s["text"]
        cur = [t.tag for t in tagger.tag(toks, task_labels=TYPE_LABELS).tags]
        g.append(to_bio(s["tags"]))
        pred[cols[0]].append(to_bio(cur))

        llm_cache = [None]  # LLM span computed once per sentence, only if needed
        def get_llm():
            if llm_cache[0] is None:
                stt = mk(toks, text); span.run(stt)
                llm_cache[0] = [t.tag for t in stt.contextual_output.sequence_output.tags]
            return llm_cache[0]

        for idx, step in enumerate(order):
            if step == "gaz":
                adds = gap_spans(toks, cur, gaz_full(toks, gaz, valid))
                conf = verify(text, adds)
                cur = apply_spans(cur, [a for a in adds if _key(a["text"]) in conf])
            elif step == "llm":
                adds = gap_spans(toks, cur, get_llm())
                conf = verify(text, adds)
                cur = apply_spans(cur, [a for a in adds if _key(a["text"]) in conf])
            elif step == "debate":
                st = mk(toks, text); st.ner_model_output = ao(toks, cur); st.contextual_output = ao(toks, get_llm())
                debate.run(st); cur = [t.tag for t in st.contextual_output.sequence_output.tags]
            elif step == "disambig":
                st = mk(toks, text); st.contextual_output = ao(toks, cur); disamb.run(st)
                cur = [t.tag for t in st.contextual_output.sequence_output.tags]
            elif step == "reflect":
                st = mk(toks, text); st.ner_model_output = ao(toks, cur); reflector.run(st)
                cur = [t.tag for t in st.contextual_output.sequence_output.tags]
            pred[cols[idx + 1]].append(to_bio(cur))

    print(f"\nPrimary: {args.model_dir}  |  order: {order}  |  Sentences: {len(sents)}  "
          f"LLM calls: {client.usage_summary()['calls']}  cost≈${client.usage_summary()['est_cost_usd']:.4f}")
    print("\n" + "=" * 66)
    print(f"{'cumulative panel':<40}{'F1':>8}{'prec':>10}{'rec':>8}")
    print("-" * 66)
    base = f1_score(g, pred[cols[0]])
    for i, c in enumerate(cols):
        f1 = f1_score(g, pred[c])
        d = "" if i == 0 else f"  ({f1-base:+.3f})"
        print(f"{c:<40}{f1:>8.3f}{precision_score(g, pred[c]):>10.3f}{recall_score(g, pred[c]):>8.3f}{d}")
    print("=" * 66)


if __name__ == "__main__":
    main()
