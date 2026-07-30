"""
annotate_ner_pool.py — gold-style entity annotation of the generated NER pool
==============================================================================
Re-annotates every sentence in the NER pool with the entities ACTUALLY PRESENT, replacing the
unreliable regex aid. Needed because a reviewer found the aid mislabelled ~12/18 workbook rows
(New York -> PER, Cairo -> OTHER, New York Fitness -> PER) and because the requested
`must_include_types` does not always describe what the generator really produced
(e.g. "PER+ORG" requested but only a person present; "Cairo University" counted as ORG+LOC).

Method: an LLM annotation pass (gpt-4.1-mini) — the same model that proved reliable at entity
recognition (6/6 on a controlled probe, incl. the obscure "Tanta"). No hardcoded gazetteers.
The annotator is told explicitly to treat NESTED names as ONE entity ("Cairo University" = ORG,
not ORG + LOC) and to ignore Arabic-script mentions (English-only policy).

Output: adds `entities_annotated` (list of {text,type}) and `types_present` (sorted list) to
        multi-agent-bert/data/NER/generated/ner_coverage_annotated.jsonl
Usage:  python experiments/switchlingua/annotate_ner_pool.py
"""
import json, os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORE = ROOT / "Modified_Version" / "core"
SRC = ROOT / "multi-agent-bert" / "data" / "NER" / "generated" / "ner_coverage_kept.jsonl"
OUT = ROOT / "multi-agent-bert" / "data" / "NER" / "generated" / "ner_coverage_annotated.jsonl"
MODEL = os.getenv("NER_ANNOTATOR_MODEL", "gpt-4.1-mini")

PROMPT = """You annotate named entities in Arabic-English code-switched sentences.

List ONLY entities written in ENGLISH/LATIN letters. Ignore Arabic-script mentions entirely.

Types:
- PER: a person's name (e.g. Sarah Hassan, Elon Musk)
- ORG: a company, university, institution, brand, team or gym (e.g. Google, Harvard, Cairo University, New York Fitness)
- LOC: a standalone city, country or place (e.g. Cairo, New York, Toronto)

CRITICAL RULES:
1. A NESTED place name inside an organisation name is NOT a separate LOC.
   "Cairo University" = ONE entity, type ORG. Do NOT also emit Cairo as LOC.
   "New York Fitness" = ONE entity, type ORG. Do NOT also emit New York as LOC.
2. "جامعة Harvard" -> the entity is "Harvard", type ORG (it is a university), not LOC.
3. Only emit an entity that literally appears in Latin letters in the sentence.
4. If there are no Latin-script entities, return an empty list.

Return STRICT JSON only:
{"entities": [{"text": "<exact span>", "type": "PER|ORG|LOC"}]}

Sentence:
"""


def main():
    import dotenv
    env = ROOT / "Modified_Version" / ".env"
    if env.exists():
        dotenv.load_dotenv(str(env), override=True)
    import ssl, httpx
    ssl._create_default_https_context = ssl._create_unverified_context
    _c = httpx.Client.__init__
    httpx.Client.__init__ = lambda s, *a, **k: (k.setdefault("verify", False), _c(s, *a, **k))[-1]
    sys.path.insert(0, str(CORE))
    import node_engine as ne
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=MODEL, temperature=0, base_url=ne.API_BASE, api_key=ne.API_KEY)
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"annotating {len(rows)} sentences with {MODEL} ...")

    out = []
    for i, r in enumerate(rows, 1):
        ents = []
        try:
            raw = llm.invoke(PROMPT + r["text"])
            s = raw.content if hasattr(raw, "content") else str(raw)
            a, b = s.find("{"), s.rfind("}")
            data = json.loads(s[a:b + 1]) if a != -1 else {}
            for e in data.get("entities", []) or []:
                t = str(e.get("type", "")).strip().upper()
                span = str(e.get("text", "")).strip()
                # keep only entities that literally appear, in Latin script
                if t in {"PER", "ORG", "LOC"} and span and span in r["text"]:
                    ents.append({"text": span, "type": t})
        except Exception as exc:
            print(f"  [{i}] annotation failed: {type(exc).__name__}")
        r["entities_annotated"] = ents
        r["types_present"] = sorted({e["type"] for e in ents})
        out.append(r)
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import collections
    combos = collections.Counter("+".join(r["types_present"]) or "(none)" for r in out)
    mismatch = sum(1 for r in out if "+".join(r["types_present"]) != r.get("group", ""))
    print(f"\nwrote {OUT}")
    print(f"ACTUAL type combinations: {dict(combos)}")
    print(f"rows where ACTUAL != REQUESTED group: {mismatch}/{len(out)}")


if __name__ == "__main__":
    main()
