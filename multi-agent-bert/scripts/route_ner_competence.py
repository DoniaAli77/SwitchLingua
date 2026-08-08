"""scripts/route_ner_competence.py

COMPETENCE-ROUTED agentic NER: the router decides *where* to intervene, an
add-only agent policy guarantees the intervention cannot damage what the primary
already got right.

Routers compared (--router):
  none        every sentence goes to the agents (the unrouted ablation baseline)
  confidence  escalate when min token confidence < --threshold (mirrors the
              classification Router in src/pipeline/router.py, default 0.6)
  competence  escalate when the primary predicts a type it is weak at
              (--weak_types, derived from DEV data — see dev_precision_by_type.py)
              or predicts nothing at all (--escalate_all_o)

Agents on escalated sentences are ADD-ONLY (gazetteer + LLM span fill the
primary's O gaps, each addition passed through LLM verify). Rewrite agents
(reflection/debate/disambiguation) are deliberately excluded: every destructive
result in the ablations came from them.

    python scripts/route_ner_competence.py --model_dir models/xlmr_gen240_ner \\
        --router competence --weak_types ORG,MISC --limit 200 \\
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
    LLMNERSpanAgent, _norm, spans_from_tags,
)
from src.agents.ner_retrieval_agents import LLMNERVerifyAgent, build_gazetteer_from_conll
from src.evaluation.ner_conll_loader import TYPE_LABELS, load_conll, tag_to_type, to_type_dataset
from src.llm.openai_client import OpenAIClient
from src.models.transformer_ner_tagger import TransformerNERTagger
from src.pipeline.ner_agentic import agentic_ner_task_config
from src.state.schema import PipelineState, StateMetadata

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
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
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
    """Spans where *cand* proposes an entity and *cur* has O — add-only by construction."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", default="models/xlmr_gen240_ner")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--env_file", default=None)
    ap.add_argument("--router", default="competence",
                    choices=["none", "confidence", "competence"])
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="confidence router: escalate when min token conf < this")
    ap.add_argument("--weak_types", default="ORG,MISC",
                    help="competence router: escalate when the primary predicts one of these")
    ap.add_argument("--escalate_all_o", action="store_true", default=True,
                    help="competence router: also escalate sentences with no predicted entity")
    ap.add_argument("--no_all_o", dest="escalate_all_o", action="store_false")
    args = ap.parse_args()

    from seqeval.metrics import f1_score, precision_score, recall_score

    train = load_conll(TRAIN)
    gaz = build_gazetteer_from_conll(train, tag_to_type)
    examples = pick_examples(train)
    sents = to_type_dataset(load_conll(TEST))[: args.limit]
    weak = {w.strip().upper() for w in args.weak_types.split(",") if w.strip()}
    valid = set(TYPE_LABELS)

    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir, device=args.device, tag_normalizer=tag_to_type)
    if not _load_env(args.env_file) or not os.environ.get("OPENAI_API_KEY"):
        print("[!] no OPENAI_API_KEY — pass --env_file"); return
    client = OpenAIClient(model=args.llm_model, max_tokens=1500)
    span = LLMNERSpanAgent(client, output_slot="contextual", descriptions=DESC, examples=examples)

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

    def should_escalate(tags, confs):
        if args.router == "none":
            return True
        if args.router == "confidence":
            return not confs or min(confs) < args.threshold
        predicted = {t for t in tags if t != "O"}
        if not predicted:
            return args.escalate_all_o
        return bool(predicted & weak)

    g, p_primary, p_routed = [], [], []
    n_esc = 0
    for s in sents:
        toks, text = s["tokens"], s["text"]
        out = tagger.tag(toks, task_labels=TYPE_LABELS)
        prim = [t.tag for t in out.tags]
        confs = [t.confidence for t in out.tags]
        g.append(to_bio(s["tags"]))
        p_primary.append(to_bio(prim))

        cur = list(prim)
        if should_escalate(prim, confs):
            n_esc += 1
            # add-only: gazetteer fills O gaps, verified
            adds = gap_spans(toks, cur, gaz_full(toks, gaz, valid))
            conf = verify(text, adds)
            cur = apply_spans(cur, [a for a in adds if _key(a["text"]) in conf])
            # add-only: LLM span fills remaining O gaps, verified
            st = mk(toks, text); span.run(st)
            llm = [t.tag for t in st.contextual_output.sequence_output.tags]
            adds = gap_spans(toks, cur, llm)
            conf = verify(text, adds)
            cur = apply_spans(cur, [a for a in adds if _key(a["text"]) in conf])
        p_routed.append(to_bio(cur))

    u = client.usage_summary()
    rule = (f"min_conf < {args.threshold}" if args.router == "confidence" else
            (f"predicts any of {sorted(weak)}" + (" or all-O" if args.escalate_all_o else "")
             if args.router == "competence" else "always"))
    print(f"\nPrimary: {args.model_dir}  |  router: {args.router} ({rule})")
    print(f"Sentences: {len(sents)}  |  escalated: {n_esc} ({n_esc/max(1,len(sents)):.0%})  "
          f"|  LLM calls: {u['calls']}  cost≈${u['est_cost_usd']:.4f}")
    print("\n" + "=" * 62)
    print(f"{'config':<34}{'F1':>8}{'prec':>10}{'rec':>8}")
    print("-" * 62)
    base = f1_score(g, p_primary)
    print(f"{'primary alone':<34}{base:>8.3f}{precision_score(g, p_primary):>10.3f}{recall_score(g, p_primary):>8.3f}")
    f1r = f1_score(g, p_routed)
    print(f"{f'{args.router}-routed + add-only':<34}{f1r:>8.3f}"
          f"{precision_score(g, p_routed):>10.3f}{recall_score(g, p_routed):>8.3f}  ({f1r-base:+.3f})")
    print("=" * 62)


if __name__ == "__main__":
    main()
