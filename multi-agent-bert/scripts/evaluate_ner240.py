"""scripts/evaluate_ner240.py

Agent-panel study on the SwitchLingua-generated NER-240 set (BIO, 4 types:
PER/LOC/ORG/MISC). Primary = OFF-THE-SHELF Davlan XLM-R (handles PER/LOC/ORG,
NO MISC). We test whether an agent panel improves on this weak primary, and
which panel is best. Entity-level F1 (seqeval, native BIO). Retrieval uses
LEAVE-ONE-OUT over the 240 (no leakage).

Panels compared:
  primary                 — Davlan alone (baseline; MISC = 0)
  dynLLM ALONE            — LLM span + dynamic few-shot, replaces primary
  primary + dynLLM+verify — LLM fills the primary's gaps (esp. MISC), verified
  primary + gaz+verify    — gazetteer (df>=2, leave-one-out) fills gaps, verified
  primary + dynLLM+gaz+verify — full panel

    python scripts/evaluate_ner240.py --env_file ../Modified_Version/.env
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.llm_ner_agent import coerce_to_valid
from src.agents.llm_ner_span_agent import (
    LLMNERSpanAgent, _norm, align_entities_to_tokens, build_span_prompt,
)
from src.agents.ner_retrieval_agents import LLMNERVerifyAgent
from src.llm.openai_client import OpenAIClient
from src.models.transformer_ner_tagger import TransformerNERTagger

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "NER" / "generated" / "expN" / "merged" / "switchlingua_ner_train_240_bio.jsonl"
LABELS = ["O", "PER", "LOC", "ORG", "MISC"]
ENT = ["PER", "LOC", "ORG", "MISC"]
DESC = {t: f"a named entity of type {t}" for t in ENT}


def to_type(tag):
    return tag.split("-", 1)[1] if "-" in tag else tag


def to_bio(type_tags):
    out, prev = [], "O"
    for t in type_tags:
        out.append("O" if t == "O" else (f"I-{t}" if t == prev else f"B-{t}"))
        prev = t
    return out


def spans_of(tokens, type_tags):
    spans, cur, buf = [], None, []
    def flush():
        if buf:
            spans.append({"text": " ".join(buf), "type": cur})
    for tok, t in zip(tokens, type_tags):
        if t == "O":
            flush(); buf.clear(); cur = None
        elif t == cur:
            buf.append(tok)
        else:
            flush(); buf.clear(); cur = t; buf.append(tok)
    flush()
    return spans


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


def _load_env(env_file):
    for c in ([env_file] if env_file else [".env", "../.env"]):
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


class Retriever:
    """IDF token-overlap over the 240; leave-one-out via text exclusion."""
    def __init__(self, rows):
        self.data, df = [], Counter()
        for r in rows:
            toks = {t for t in (_norm(x) for x in r["tokens"]) if t}
            self.data.append({"text": r["text"], "toks": toks,
                              "ents": spans_of(r["tokens"], [to_type(t) for t in r["ner_tags"]])})
            for t in toks:
                df[t] += 1
        n = max(1, len(self.data))
        self.idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}

    def top(self, tokens, k, exclude_text):
        q = {t for t in (_norm(x) for x in tokens) if t}
        scored = [(sum(self.idf.get(t, 0.0) for t in (q & d["toks"])), d)
                  for d in self.data if d["text"] != exclude_text and (q & d["toks"])]
        scored.sort(key=lambda x: -x[0])
        return [(d["text"], d["ents"]) for _, d in scored[:k]]


def build_gazetteer_lofo(rows, min_df=2):
    """Gazetteer keeping only entities appearing in >= min_df sentences (LOO-safe)."""
    typ, sent = defaultdict(Counter), defaultdict(set)
    for i, r in enumerate(rows):
        for e in spans_of(r["tokens"], [to_type(t) for t in r["ner_tags"]]):
            key = " ".join(_norm(x) for x in e["text"].split() if _norm(x))
            if key:
                typ[key][e["type"]] += 1; sent[key].add(i)
    return {k: typ[k].most_common(1)[0][0] for k in typ if len(sent[k]) >= min_df}


def gaz_tags(tokens, gaz):
    norm = [_norm(t) for t in tokens]
    tags = ["O"] * len(tokens)
    maxlen = max((len(k.split()) for k in gaz), default=1)
    i = 0
    while i < len(tokens):
        hit = False
        for L in range(min(maxlen, len(tokens) - i), 0, -1):
            key = " ".join(norm[i:i + L])
            if key in gaz:
                for j in range(i, i + L):
                    tags[j] = gaz[key]
                i += L; hit = True; break
        if not hit:
            i += 1
    return tags


def _key(t):
    return " ".join(_norm(x) for x in t.split())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Davlan/xlm-roberta-base-ner-hrl")
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--env_file", default=None)
    ap.add_argument("--no_llm", action="store_true", help="deterministic only (primary + gazetteer)")
    args = ap.parse_args()

    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    rows = [json.loads(l) for l in DATA.open(encoding="utf-8")]
    print(f"NER-240: {len(rows)} sentences.")
    retr = Retriever(rows)
    gaz = build_gazetteer_lofo(rows)
    print(f"Gazetteer (df>=2, leave-one-out safe): {len(gaz)} entries.")

    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir or args.model, device=args.device, tag_normalizer=to_type)
    valid = set(LABELS)
    client = None
    if not args.no_llm:
        if not _load_env(args.env_file) or not os.environ.get("OPENAI_API_KEY"):
            print("[!] no OPENAI_API_KEY — pass --env_file (or use --no_llm)"); return
        client = OpenAIClient(model=args.llm_model, max_tokens=1500)

    if args.no_llm:
        cfgs = ["primary", "prim+gaz(no verify)"]
    else:
        cfgs = ["primary", "dynLLM_alone", "prim+dyn+verify", "prim+gaz+verify", "prim+dyn+gaz+verify"]
    g_bio = []
    pred = {c: [] for c in cfgs}
    for r in rows:
        toks = r["tokens"]
        gold_t = [to_type(t) for t in r["ner_tags"]]
        g_bio.append(to_bio(gold_t))
        prim = [t.tag for t in tagger.tag(toks, task_labels=LABELS).tags]
        gaz_adds = gap_spans(toks, prim, gaz_tags(toks, gaz))

        if args.no_llm:
            pred["primary"].append(to_bio(prim))
            pred["prim+gaz(no verify)"].append(to_bio(apply_spans(prim, gaz_adds)))
            continue

        ex = retr.top(toks, args.k, exclude_text=r["text"])
        raw = client.generate(build_span_prompt("ner", LABELS, toks, r["text"], DESC, ex))
        try:
            ents, _ = LLMNERSpanAgent._parse(raw)
        except Exception:
            ents = []
        llm_full = align_entities_to_tokens(toks, ents, valid)
        dyn_adds = gap_spans(toks, prim, llm_full)

        union = dyn_adds + [a for a in gaz_adds
                            if not any(a["start"] == d["start"] and a["end"] == d["end"] for d in dyn_adds)]
        conf = set()
        if union:
            vraw = client.generate(LLMNERVerifyAgent._prompt(r["text"], union))
            try:
                conf = {_key(e["text"]) for e in LLMNERVerifyAgent._parse_kept(vraw, union)}
            except Exception:
                conf = {_key(a["text"]) for a in union}
        v = lambda spans: [a for a in spans if _key(a["text"]) in conf]

        pred["primary"].append(to_bio(prim))
        pred["dynLLM_alone"].append(to_bio(llm_full))
        pred["prim+dyn+verify"].append(to_bio(apply_spans(prim, v(dyn_adds))))
        pred["prim+gaz+verify"].append(to_bio(apply_spans(prim, v(gaz_adds))))
        pred["prim+dyn+gaz+verify"].append(to_bio(apply_spans(prim, v(dyn_adds + gaz_adds))))

    if client is not None:
        print(f"\nLLM calls: {client.usage_summary()['calls']}  cost≈${client.usage_summary()['est_cost_usd']:.4f}")
    print("\n" + "=" * 60)
    print(f"{'panel':<24}{'F1':>8}{'precision':>11}{'recall':>9}")
    print("-" * 60)
    scored = []
    for c in cfgs:
        f1 = f1_score(g_bio, pred[c]); p = precision_score(g_bio, pred[c]); rc = recall_score(g_bio, pred[c])
        scored.append((c, f1)); print(f"{c:<24}{f1:>8.3f}{p:>11.3f}{rc:>9.3f}")
    print("=" * 60)
    best = max(scored, key=lambda x: x[1])
    base = dict(scored)["primary"]
    print(f"BEST panel: '{best[0]}' (F1={best[1]:.3f})  |  improvement over primary: {best[1]-base:+.3f}")
    print("\nPer-type for the BEST panel:")
    print(classification_report(g_bio, pred[best[0]], digits=3))


if __name__ == "__main__":
    main()
