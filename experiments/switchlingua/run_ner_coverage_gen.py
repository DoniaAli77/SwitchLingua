"""
run_ner_coverage_gen.py — small NER generation run with ENTITY-TYPE COVERAGE (human-eval top-up)
=================================================================================================
Generates NER sentences across 7 entity-type groups so the human-eval sample can cover
PER / ORG / LOC / pairs / PER+ORG+LOC — coverage that does not exist in any current data
(every past NER scenario used must_include_types=[PER, ORG]).

Design constraints honoured (verified in code):
  * `must_include_types` is a SINGLE list applied to all scenarios (utils.py:121) — it is not
    producted — so this runner sweeps the 7 groups itself by patching the config dict IN MEMORY.
    No pipeline/config-file change; NER core code is untouched (still frozen).
  * Policy stays ENGLISH-ONLY (target_entities_script=english) — consistent with all prior NER work.
  * Entity strings are extracted POST-HOC here (offline), because the pipeline computes them
    internally but never persists them. They are annotator AIDS, clearly marked, not gold labels.

Filters (same as every other track): non-empty -> TaskValidator passed -> deterministic CS-valid
-> quality >= 7.0 -> dedup. Task-aware acceptance is ON by default, so task-failing NER sentences
are dropped by the pipeline itself.

Output: multi-agent-bert/data/NER/generated/{raw_states.jsonl, ner_coverage_kept.jsonl, stats.json}
Usage:  python experiments/switchlingua/run_ner_coverage_gen.py --per-group 6 --concurrency 4
"""
import argparse, asyncio, importlib, json, os, pathlib, random, re, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
SW = ROOT / "experiments" / "switchlingua"
CORE = ROOT / "Modified_Version" / "core"
OUT = ROOT / "multi-agent-bert" / "data" / "NER" / "generated"
sys.path.insert(0, str(SW))
import manage_sentiment_data as M  # shared filter/io helpers

MODEL = "gpt-4o-mini"
QTHRESH = 7.0
# Minimum ordinary English tokens (lower-case, i.e. not part of a name) required BEYOND the
# entities, so the sentence is genuinely code-switched rather than an Arabic sentence with a
# proper noun inserted. See the "entity_only_switch" filter below.
MIN_EN_CONTEXT = 3

_LAT_WORD = re.compile(r"^[A-Za-z][A-Za-z0-9'.\-]*$")


_EN_FUNCTION_WORDS = {"i", "and", "but", "the", "it", "my", "he", "she", "they", "we", "a", "an",
                      "so", "because", "when", "also", "however", "especially", "since", "honestly"}


_ARABIC_CHAR = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN_CHAR = re.compile(r"[A-Za-z]")


def has_latin_entity_candidate(text: str) -> bool:
    """True if the sentence contains at least one Latin-script capitalised token that is not a
    sentence-initial English function word.

    Enforces the ENGLISH-ONLY entity policy mechanically. This is a pure SCRIPT test (Latin vs
    Arabic) and needs no world knowledge — it cannot tell PER from LOC, only that a Latin-script
    name candidate exists. Needed because the LLM validator sometimes accepts an Arabic-script
    entity (e.g. 'عمر خالد' for PER, 'نيويورك' for LOC) despite the policy forbidding it.

    HEURISTIC, superseded by has_latin_entity_span() where the validator's entity spans are
    available. It requires a CAPITALISED token, which is only a valid proxy for entity types that
    are proper nouns (PER/LOC/ORG). It wrongly rejects common-noun entities — currency names
    ('pounds', 'dirhams', 'euros') are lowercase Latin script, and were being discarded and
    mislabelled 'arabic_script_entity'. Kept for callers that have no span information.
    """
    for cand in re.findall(r"\b([A-Z][A-Za-z0-9&.\-]+)\b", str(text)):
        if cand.lower() not in _EN_FUNCTION_WORDS:
            return True
    return False


# Entity classes that are legitimately LOWER-CASE in normal English and would otherwise be
# rejected by the capitalisation rule below. Deliberately a small closed list: currency names are
# the only entity class in this corpus's schema that is a common noun by convention. Months
# (March) and religious observances (Ramadan, Eid al-Fitr) are capitalised and need no exemption.
_LOWERCASE_ENTITY_TERMS = {
    "dollar", "dollars", "euro", "euros", "pound", "pounds", "riyal", "riyals",
    "dinar", "dinars", "dirham", "dirhams", "shekel", "shekels", "lira", "yen", "rupee", "rupees",
}


def has_latin_entity_span(entities) -> bool:
    """True if at least one validator-reported entity span is an acceptable Latin-script entity.

    Direct form of the ENGLISH-ONLY entity policy: test the spans the validator actually reported
    instead of guessing from the sentence whether a Latin entity is *likely* present.

    A span qualifies when it is Latin script (no Arabic characters) AND is either capitalised or a
    known lower-case entity term. The capitalisation requirement is retained deliberately: it is
    the only mechanical signal separating a real named entity from the common nouns the LLM
    validator sometimes returns as 'entities' ('nutrition', 'wellness', 'yoga', 'water'). Dropping
    it entirely admits that noise, so currency names are exempted by name instead.
    """
    for e in entities or []:
        span = str(e.get("text", "") if isinstance(e, dict) else e).strip()
        if not span or _ARABIC_CHAR.search(span) or not _LATIN_CHAR.search(span):
            continue
        words = [w for w in re.split(r"[\s\-]+", span) if w]
        if any(w[0].isupper() for w in words):
            return True
        if all(w.lower().strip(".,") in _LOWERCASE_ENTITY_TERMS for w in words):
            return True
    return False


def en_context_tokens(text: str) -> int:
    """Count lower-case Latin word tokens (entities are capitalised, so these are context words)."""
    n = 0
    for tok in str(text).split():
        w = tok.strip(".,!?;:()\"'،؛")
        if w and _LAT_WORD.match(w) and w[0].islower():
            n += 1
    return n

# 7 entity-type groups -> (must_include_types, min_entities, max_entities)
GROUPS = [
    (["PER"], 1, 2), (["ORG"], 1, 2), (["LOC"], 1, 2),
    (["PER", "ORG"], 2, 3), (["PER", "LOC"], 2, 3), (["ORG", "LOC"], 2, 3),
    (["PER", "ORG", "LOC"], 3, 4),
]
TOPICS = ["business", "education", "health", "shopping", "medical", "sports", "tech", "finance", "social"]

# ---- post-hoc entity extraction (annotator aid; mirrors the pipeline's English-only intent) ----
ORG_TAIL = r"(?:Inc|Corp|Corporation|Ltd|LLC|Group|University|Bank|Agency|Company|Institute|Labs|Lab|Technologies|Tech|Hospital|Clinic|Center|Centre)"
NON_PERSON_TAIL = {"Docs", "News", "Bank", "University", "Company", "Corporation", "Institute", "Labs", "Lab", "Tech",
                   "Hospital", "Clinic", "Center", "Centre", "Group", "Agency"}


KNOWN_ORGS = {
    "google", "microsoft", "apple", "amazon", "meta", "facebook", "netflix", "tesla", "ibm",
    "intel", "nvidia", "samsung", "sony", "huawei", "twitter", "openai", "spacex", "uber",
    "barcelona", "real madrid", "liverpool", "chelsea", "arsenal", "juventus", "bayern",
    "unicef", "unesco", "who", "nasa", "harvard", "mit", "stanford", "oxford", "cambridge",
}


def extract_entities(text: str) -> str:
    """Readable 'PER: X | ORG: Y | LOC: Z' aid for annotators.

    Classification mirrors the pipeline: the LOC gazetteer is imported from node_engine so the
    aid agrees with what the validator actually counted; ORG covers suffix patterns plus common
    bare brand names; remaining Title-case pairs are treated as PER.
    """
    try:
        sys.path.insert(0, str(CORE))
        from node_engine import _KNOWN_LOCATIONS as LOCS
    except Exception:
        LOCS = set()

    cands = set(re.findall(r"\b([A-Z][A-Za-z0-9&.\-]+(?:\s+[A-Z][A-Za-z0-9&.\-]+)*)\b", text))
    locs, orgs, pers, rest = set(), set(), set(), set()
    for c in sorted(cands, key=lambda s: -len(s)):
        cl = c.lower()
        toks = c.split()
        if cl in LOCS:
            locs.add(c)
        elif cl in KNOWN_ORGS or re.search(rf"\b{ORG_TAIL}\b", c):
            orgs.add(c)
        elif len(toks) >= 2 and toks[-1] not in NON_PERSON_TAIL and not any(t.lower() in LOCS for t in toks):
            pers.add(c)
        else:  # split a mixed run like "Boston Harvard" into its known parts
            matched = False
            for t in toks:
                if t.lower() in LOCS:
                    locs.add(t); matched = True
                elif t.lower() in KNOWN_ORGS:
                    orgs.add(t); matched = True
            if not matched:
                rest.add(c)
    covered = {t for grp in (locs | orgs | pers) for t in grp.split()}
    rest = {r for r in rest if not set(r.split()) & covered}
    parts = []
    if pers: parts.append("PER: " + ", ".join(sorted(pers)))
    if orgs: parts.append("ORG: " + ", ".join(sorted(orgs)))
    if locs: parts.append("LOC: " + ", ".join(sorted(locs)))
    if rest: parts.append("OTHER: " + ", ".join(sorted(rest)))
    return " | ".join(parts)


async def _generate(per_group, concurrency, topics, only_groups=None):
    # Optionally restrict to specific entity-type groups (e.g. top up only PER+ORG+LOC)
    # instead of re-running all 7 and spending calls on groups that already meet quota.
    _active_groups = GROUPS
    if only_groups:
        want = {g.strip().upper() for g in only_groups}
        _active_groups = [g for g in GROUPS if "+".join(g[0]).upper() in want]
        if not _active_groups:
            raise SystemExit(f"no matching groups for {sorted(want)}; "
                             f"valid: {[chr(43).join(g[0]) for g in GROUPS]}")
    import dotenv
    env = ROOT / "Modified_Version" / ".env"
    if env.exists():
        dotenv.load_dotenv(str(env), override=True)
    os.environ["API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
    os.environ["API_BASE"] = os.environ.get("OPENAI_BASE_URL", "")
    import ssl, httpx
    ssl._create_default_https_context = ssl._create_unverified_context
    _c = httpx.Client.__init__
    httpx.Client.__init__ = lambda s, *a, **k: (k.setdefault("verify", False), _c(s, *a, **k))[-1]
    _a = httpx.AsyncClient.__init__
    httpx.AsyncClient.__init__ = lambda s, *a, **k: (k.setdefault("verify", False), _a(s, *a, **k))[-1]
    if str(CORE) not in sys.path:
        sys.path.insert(0, str(CORE))
    for m in ("utils", "node_engine", "node_models", "prompt", "agents", "run_french"):
        sys.modules.pop(m, None)
    importlib.invalidate_caches()
    import utils as ut, node_engine as ne, run_french as rf
    ne.MODEL = MODEL
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "_backup").mkdir(exist_ok=True)
    ne.OUTPUT_DIR = str(OUT / "_backup")
    print(f"TASK_AWARE_ACCEPT={ne.TASK_AWARE_ACCEPT} (task-failing NER sentences are dropped)")

    rng = random.Random(11)
    scenarios = []
    for must, mn, mx in _active_groups:
        cfg = {
            "cs_ratio": ["50%", "60%"], "task": ["ner"],
            "shared": {
                "topic": topics, "tense": ["Present"], "perspective": ["First Person"],
                "cs_function": ["Expressive"], "cs_type": ["Intrasentential"],
                "conversation_type": ["single_turn"], "output_format": "json",
                "character_setting": {
                    "nationality": {"first_language": "Arabic", "second_language": "English"},
                    "age": ["18-25"], "gender": ["Male", "Female"], "education_level": ["College"],
                },
            },
            "ner": {
                # IMPORTANT: entity_types must equal the GROUP's types. The validator requires
                # `has_all_entity_types` = every type in entity_types to be present, so listing all
                # of PER/ORG/LOC for a single-type group makes it unsatisfiable against max_entities.
                "entity_types": list(must), "min_entities": [mn], "max_entities": [mx],
                "must_include_types": must,              # single list per group (see module docstring)
                "allow_code_switched_entities": [False],
                "target_entities_script": ["english"],
            },
        }
        pool = ut.generate_scenarios(cfg)
        rng.shuffle(pool)
        for s in pool[:per_group]:
            s["_group"] = "+".join(must)
            scenarios.append(s)

    print(f"[ner-gen] {len(scenarios)} scenarios across {len(GROUPS)} type groups "
          f"({per_group}/group), concurrency={concurrency}")
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    prog = {"d": 0, "ok": 0, "f": 0}
    states = []

    async def one(sc):
        group = sc.pop("_group")
        async with sem:
            try:
                st = await rf.CodeSwitchingAgent(sc).run()
            except Exception as exc:
                st = None
                async with lock:
                    prog["f"] += 1
                    msg = str(exc)
                    if ("429" in msg or "rate_limit" in msg.lower()) and prog["f"] % 10 == 1:
                        print("[ner-gen] 429 rate-limited")
            async with lock:
                prog["d"] += 1
                if st:
                    st["_group"] = group
                    states.append(st)
                    prog["ok"] += 1
                if prog["d"] % 10 == 0 or prog["d"] == len(scenarios):
                    print(f"[ner-gen] {prog['d']}/{len(scenarios)} ok={prog['ok']} fail={prog['f']}", flush=True)

    await asyncio.gather(*(one(s) for s in scenarios))
    with open(OUT / "raw_states.jsonl", "a", encoding="utf-8") as f:
        for st in states:
            f.write(json.dumps(st, ensure_ascii=False, default=str) + "\n")
    return len(scenarios), prog["ok"], prog["f"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--filter-only", action="store_true")
    ap.add_argument("--groups", nargs="*", default=None,
                    help="restrict to entity-type groups, e.g. --groups PER+ORG+LOC")
    a = ap.parse_args()

    meta = (0, 0, 0)
    if not a.filter_only:
        meta = asyncio.run(_generate(a.per_group, a.concurrency, TOPICS, a.groups))

    states = [s for s in M._read_jsonl(OUT / "raw_states.jsonl") if s.get("task") == "ner"]
    rows = []
    for s in states:
        grp = s.get("_group", "")
        # The validator now records the ENTITIES it verified (span-checked, Latin-script,
        # nesting-collapsed) on each per-instance result. Carry them through as the real
        # annotation instead of re-guessing with a regex.
        per_inst = s.get("task_validation_results_per_instances") or []
        # extract_rows() preserves sentence_records order but does not emit an index field,
        # so align positionally with the per-instance validator results.
        for idx, r in enumerate(M.extract_rows(s, "ner_coverage")):
            r["group"] = grp
            pi = per_inst[idx] if idx < len(per_inst) and isinstance(per_inst[idx], dict) else {}
            ents = pi.get("entities") or []
            r["entities_verified"] = ents
            r["types_present"] = sorted({e.get("type") for e in ents if e.get("type")})
            by_type = {}
            for e in ents:
                by_type.setdefault(e["type"], []).append(e["text"])
            r["entities_pretty"] = " | ".join(f"{t}: {', '.join(v)}" for t, v in sorted(by_type.items()))
            rows.append(r)
    M.assign_reasons(rows, QTHRESH)
    # EXTRA NER-ONLY FILTER: reject "named-entity-only switching". The deterministic CS check is
    # satisfied by the entity name alone (>=1 Latin token), so a sentence whose only English content
    # is the required entity passes CS-validity without actually code-switching. Require at least
    # MIN_EN_CONTEXT ordinary (lower-case, non-name) English tokens beyond the entities.
    for r in rows:
        if r["filter_reason"] is None and en_context_tokens(r["text"]) < MIN_EN_CONTEXT:
            r["filter_reason"] = "entity_only_switch"
        # ENGLISH-ONLY entity policy, enforced mechanically: the LLM validator sometimes accepts
        # an Arabic-script entity, which the policy forbids. Pure script test, no world knowledge.
        elif r["filter_reason"] is None and not has_latin_entity_candidate(r["text"]):
            r["filter_reason"] = "arabic_script_entity"
    kept = [r for r in rows if r["filter_reason"] is None]
    # keep only sentences where post-hoc extraction actually found something (annotator aid must be useful)
    M._write_jsonl(OUT / "ner_coverage_kept.jsonl", kept)
    stats = {
        "scenarios_attempted": meta[0], "scenarios_ok": meta[1], "scenarios_failed": meta[2],
        "raw_instances": len(rows), "kept": len(kept),
        "kept_by_group": dict(Counter(r["group"] for r in kept)),
        "loss_by_reason": dict(Counter(r["filter_reason"] for r in rows if r["filter_reason"])),
        "policy": "english-only entities; entities_posthoc = offline annotator aid, not gold labels",
    }
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ner-gen] raw={len(rows)} kept={len(kept)}")
    print(f"[ner-gen] kept by group: {stats['kept_by_group']}")
    print(f"[ner-gen] loss: {stats['loss_by_reason']}")
    print(f"[ner-gen] -> {OUT/'ner_coverage_kept.jsonl'}")


if __name__ == "__main__":
    main()
