"""scripts/evaluate_ner_retrieve_verify.py

Last principled lever: PRIMARY + gazetteer RETRIEVE-then-VERIFY.
Keep every primary prediction; the gazetteer proposes entities only in the
primary's O gaps; the LLM then VERIFIES just those proposals and we keep only the
confirmed ones. Aim: the recall gain of retrieval WITHOUT the precision loss.

Compares (full test, entity-level seqeval):
  PRIMARY            — baseline 0.816
  PRIMARY + gaz      — augment, no verify (adds false positives)
  PRIMARY + gaz + VERIFY  — additions filtered by the LLM

    python scripts/evaluate_ner_retrieve_verify.py --model_dir models/xlmr_sabty_ner \\
        --limit 300 --env_file ../Modified_Version/.env
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
from src.agents.llm_ner_span_agent import _norm
from src.agents.ner_retrieval_agents import LLMNERVerifyAgent, build_gazetteer_from_conll
from src.evaluation.ner_conll_loader import TYPE_LABELS, load_conll, tag_to_type, to_type_dataset
from src.llm.openai_client import OpenAIClient
from src.models.transformer_ner_tagger import TransformerNERTagger

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "NER" / "Train_AR-EN_NER.txt"
TEST = ROOT / "data" / "NER" / "Test_AR-EN_NER.txt"


def to_bio(type_tags):
    out, prev = [], "O"
    for t in type_tags:
        out.append("O" if t == "O" else (f"I-{t}" if t == prev else f"B-{t}"))
        prev = t
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


def gaz_matches(tokens, prim, gazetteer, valid):
    """Return gazetteer additions (in primary's O gaps) as span dicts w/ positions."""
    norm = [_norm(t) for t in tokens]
    maxlen = max((len(k.split()) for k in gazetteer), default=1)
    spans, i = [], 0
    while i < len(tokens):
        hit = False
        for L in range(min(maxlen, len(tokens) - i), 0, -1):
            if any(prim[i + j] != "O" for j in range(L)):
                continue
            key = " ".join(norm[i:i + L])
            if key in gazetteer:
                ty = coerce_to_valid(gazetteer[key], valid)
                if ty != "O":
                    spans.append({"start": i, "end": i + L, "type": ty,
                                  "text": " ".join(tokens[i:i + L])})
                    i += L; hit = True; break
        if not hit:
            i += 1
    return spans


def apply_spans(prim, spans):
    out = list(prim)
    for s in spans:
        for j in range(s["start"], s["end"]):
            out[j] = s["type"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", default="models/xlmr_sabty_ner")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--env_file", default=None)
    args = ap.parse_args()

    from seqeval.metrics import f1_score, precision_score, recall_score

    gazetteer = build_gazetteer_from_conll(load_conll(TRAIN), tag_to_type)
    sents = to_type_dataset(load_conll(TEST))
    if args.limit:
        sents = sents[:args.limit]
    tagger = TransformerNERTagger.from_pretrained(
        checkpoint=args.model_dir, device=args.device, tag_normalizer=tag_to_type)
    if not _load_env(args.env_file) or not os.environ.get("OPENAI_API_KEY"):
        print("[!] no OPENAI_API_KEY — pass --env_file"); return
    client = OpenAIClient(model=args.llm_model, max_tokens=1500)
    valid = set(TYPE_LABELS)

    g_bio, base_bio, aug_bio, ver_bio = [], [], [], []
    for s in sents:
        prim = [tag_to_type(t.tag) for t in tagger.tag(s["tokens"], task_labels=None).tags]
        n = min(len(s["tags"]), len(prim))
        gold, prim, toks = s["tags"][:n], prim[:n], s["tokens"][:n]
        adds = gaz_matches(toks, prim, gazetteer, valid)

        aug = apply_spans(prim, adds)                       # retrieve (no verify)
        if adds:
            raw = client.generate(LLMNERVerifyAgent._prompt(s["text"], adds))
            try:
                kept = LLMNERVerifyAgent._parse_kept(raw, adds)
                kept_norm = {" ".join(_norm(x) for x in e["text"].split()) for e in kept}
            except Exception:
                kept_norm = {" ".join(_norm(x) for x in a["text"].split()) for a in adds}
            confirmed = [a for a in adds
                         if " ".join(_norm(x) for x in a["text"].split()) in kept_norm]
            ver = apply_spans(prim, confirmed)              # retrieve + verify
        else:
            ver = prim

        g_bio.append(to_bio(gold)); base_bio.append(to_bio(prim))
        aug_bio.append(to_bio(aug)); ver_bio.append(to_bio(ver))

    def line(name, pred):
        print(f"  {name:<26} F1={f1_score(g_bio, pred):.3f}  "
              f"P={precision_score(g_bio, pred):.3f}  R={recall_score(g_bio, pred):.3f}")

    print(f"\nSentences: {len(sents)}  LLM calls: {client.usage_summary()['calls']}"
          f"  cost≈${client.usage_summary()['est_cost_usd']:.4f}")
    print("\n=== entity-level (seqeval) ===")
    line("PRIMARY (baseline)", base_bio)
    line("PRIMARY + gaz", aug_bio)
    line("PRIMARY + gaz + VERIFY", ver_bio)


if __name__ == "__main__":
    main()
