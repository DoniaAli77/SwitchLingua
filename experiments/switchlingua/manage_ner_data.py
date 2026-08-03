"""
manage_ner_data.py — accumulation workflow for the NER 240/480/960 ladder (Experiment N)
=========================================================================================
Same resume-safe pattern as manage_sentiment_data.py / manage_topic_data.py, adapted to NER:

  generate : resume-aware run over entity-type GROUPS (must_include_types is a single list per
             scenario batch, so the runner sweeps the groups itself), append raw states, filter.
  merge    : cross-dedup + select to a target size using CORPUS-MATCHED group proportions
             (Latin-script shares measured on the Sabty AR-EN NER corpus).
  bio      : export token-level BIO tags (needed to TRAIN a NER model; the pool stores spans).

Filters (identical to the human-eval pool, all verified):
  non-empty -> TaskValidator (EVIDENCE-based: verdict recomputed in code from the entities the
  validator reports) -> deterministic CS-valid -> quality >= 7.0 -> >=3 ordinary English words
  beyond the entity names (entity_only_switch) -> Latin-script entity present -> dedup.

Layout: multi-agent-bert/data/NER/generated/expN/
        daily_runs/run_YYYYMMDD_{raw,filtered}.jsonl, completed_scenarios_ner.json,
        merged/switchlingua_ner_train_{N}.{jsonl,csv}, merged/*_bio.jsonl

Usage:
  python experiments/switchlingua/manage_ner_data.py generate --max 120 --concurrency 4
  python experiments/switchlingua/manage_ner_data.py merge --target 240
  python experiments/switchlingua/manage_ner_data.py bio --target 240
"""
import argparse, asyncio, datetime, importlib, json, os, pathlib, random, re, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
SW = ROOT / "experiments" / "switchlingua"
CORE = ROOT / "Modified_Version" / "core"
sys.path.insert(0, str(SW))
import manage_sentiment_data as M
from run_ner_coverage_gen import en_context_tokens, has_latin_entity_candidate, MIN_EN_CONTEXT

CONFIG = SW / "config_ner_expN_v1.yaml"
BASE = ROOT / "multi-agent-bert" / "data" / "NER" / "generated" / "expN"
DAILY = BASE / "daily_runs"
MERGED = BASE / "merged"
MANIFEST = BASE / "completed_scenarios_ner.json"
MODEL = "gpt-4o-mini"
QTHRESH = 7.0

# Entity-type groups + their share of LATIN-SCRIPT entity-bearing sentences in the Sabty corpus.
# (must_include_types, min_entities, max_entities, corpus_share)
#
# MISC+ORG WAS DROPPED (measured, not assumed): 71 dedicated scenarios produced ZERO usable
# sentences — the same failure seen for PER+ORG+LOC (23 scenarios, zero). Any group requiring ORG
# *plus* another type in one sentence with genuine code-switching consistently fails: the model
# trades entity coverage against real code-switching. Its 1.5% corpus share is renormalised across
# the remaining groups below. Measured per-group yields (kept/scenario) for reference:
#   ORG 2.02 | MISC 1.95 | PER 1.09 | LOC 0.85 | PER+ORG 0.76 | PER+LOC 0.23 | MISC+ORG 0.00
# PER+LOC is kept despite its low yield, but is CAPPED by availability at merge time (the merge
# tops up any shortfall from the other groups rather than spending ~570 scenarios on one group).
GROUPS = [
    (["PER"],          1, 2, 0.329),
    (["LOC"],          1, 2, 0.284),
    (["PER", "LOC"],   2, 3, 0.146),
    (["MISC"],         1, 2, 0.146),
    (["ORG"],          1, 2, 0.080),
    (["PER", "ORG"],   2, 3, 0.014),
]
GROUP_KEY = {"+".join(g[0]): g for g in GROUPS}


def _manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"completed": {}}


def _save_manifest(m):
    m["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def _scenarios(config_path=None, only_groups=None):
    """Build every (group x scenario) combination, tagged with its group.
    `only_groups` restricts generation to the groups that are still short of their corpus-matched
    target, so no calls are spent on groups that already exceed even the 960 quota."""
    if str(CORE) not in sys.path:
        sys.path.insert(0, str(CORE))
    sys.modules.pop("utils", None)
    import utils as ut
    cfg = ut.load_config(str(config_path or CONFIG))
    want = {g.strip().upper() for g in (only_groups or [])}
    out = []
    for must, mn, mx, _share in GROUPS:
        if want and "+".join(must).upper() not in want:
            continue
        c = json.loads(json.dumps(cfg["pre_execute"]))       # deep copy
        c["ner"]["entity_types"] = list(must)                # validator needs ALL of entity_types
        c["ner"]["must_include_types"] = list(must)
        c["ner"]["min_entities"] = [mn]
        c["ner"]["max_entities"] = [mx]
        for s in ut.generate_scenarios(c):
            if s.get("task") != "ner":
                continue
            s["_group"] = "+".join(must)
            s["scenario_id"] = M._scenario_id(s) + ":" + s["_group"]
            out.append(s)
    return out


# --------------------------------------------------------------------------- generate
def cmd_generate(a):
    date = a.date or datetime.date.today().strftime("%Y%m%d")
    raw_path = DAILY / f"run_{date}_raw.jsonl"
    scen = _scenarios(a.config, a.groups)
    m = _manifest()
    done = set(m["completed"])
    todo = [s for s in scen if s["scenario_id"] not in done]
    random.seed(a.seed); random.shuffle(todo)
    attempted = todo[: a.max] if a.max else todo
    print(f"[ner] space={len(scen)} done={len(done)} remaining={len(todo)} attempting={len(attempted)}")
    if not attempted:
        print("[ner] nothing to do — run `merge`."); return

    import dotenv
    env = ROOT / "Modified_Version" / ".env"
    if env.exists():
        dotenv.load_dotenv(str(env), override=True)
    os.environ["API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
    os.environ["API_BASE"] = os.environ.get("OPENAI_BASE_URL", "")
    import ssl, httpx
    ssl._create_default_https_context = ssl._create_unverified_context
    _c = httpx.Client.__init__
    httpx.Client.__init__ = lambda s, *x, **k: (k.setdefault("verify", False), _c(s, *x, **k))[-1]
    _a = httpx.AsyncClient.__init__
    httpx.AsyncClient.__init__ = lambda s, *x, **k: (k.setdefault("verify", False), _a(s, *x, **k))[-1]
    if str(CORE) not in sys.path:
        sys.path.insert(0, str(CORE))
    for mod in ("utils", "node_engine", "node_models", "prompt", "agents", "run_french"):
        sys.modules.pop(mod, None)
    importlib.invalidate_caches()
    import node_engine as ne, run_french as rf
    ne.MODEL = MODEL
    (BASE / "_backup").mkdir(parents=True, exist_ok=True)
    ne.OUTPUT_DIR = str(BASE / "_backup")
    DAILY.mkdir(parents=True, exist_ok=True)
    print(f"[ner] gen={MODEL} ner_validator={ne.NER_VALIDATOR_MODEL} TASK_AWARE_ACCEPT={ne.TASK_AWARE_ACCEPT}")

    sem = asyncio.Semaphore(a.concurrency); lock = asyncio.Lock()
    prog = {"d": 0, "ok": 0, "f": 0}

    async def one(sc):
        grp = sc.pop("_group"); sid = sc["scenario_id"]
        async with sem:
            try:
                st = await rf.CodeSwitchingAgent(sc).run()
            except Exception as exc:
                st = None
                async with lock:
                    prog["f"] += 1
                    msg = str(exc)
                    if ("429" in msg or "rate_limit" in msg.lower()) and prog["f"] % 20 == 1:
                        print("[ner] 429 rate-limited")
            async with lock:
                prog["d"] += 1
                if st:
                    st["_group"] = grp; st["scenario_id"] = sid
                    M._write_jsonl(raw_path, [st], append=True)
                    m["completed"][sid] = date; prog["ok"] += 1
                    if prog["ok"] % 20 == 0:
                        _save_manifest(m)
                if prog["d"] % 20 == 0 or prog["d"] == len(attempted):
                    print(f"[ner] {prog['d']}/{len(attempted)} ok={prog['ok']} fail={prog['f']}", flush=True)

    asyncio.run(_gather(attempted, one))
    _save_manifest(m)

    rows = _filter_rows(M._read_jsonl(raw_path), a.quality_threshold)
    kept = [r for r in rows if r["filter_reason"] is None]
    M._write_jsonl(DAILY / f"run_{date}_filtered.jsonl", kept)
    stats = {"date": date, "attempted": len(attempted), "ok": prog["ok"], "failed": prog["f"],
             "raw_instances": len(rows), "kept": len(kept),
             "kept_by_group": dict(Counter(r["group"] for r in kept)),
             "loss": dict(Counter(r["filter_reason"] for r in rows if r["filter_reason"]))}
    (DAILY / f"run_{date}_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ner] day {date}: ok={prog['ok']} raw={len(rows)} kept={len(kept)} loss={stats['loss']}")
    print(f"[ner] kept by group: {stats['kept_by_group']}")
    if prog["ok"]:
        print(f"[ner] yield = {len(kept)/prog['ok']:.2f} kept/scenario | manifest {len(m['completed'])}/{len(scen)}")


async def _gather(items, coro):
    await asyncio.gather(*(coro(x) for x in items))


def _filter_rows(states, qthresh):
    """Same filter chain as the verified human-eval pool, incl. the validator's own entities."""
    rows = []
    for s in states:
        if s.get("task") != "ner":
            continue
        grp = s.get("_group", "")
        per_inst = s.get("task_validation_results_per_instances") or []
        for idx, r in enumerate(M.extract_rows(s, "ner_expN")):
            pi = per_inst[idx] if idx < len(per_inst) and isinstance(per_inst[idx], dict) else {}
            ents = pi.get("entities") or []
            r["group"] = grp
            r["entities_verified"] = ents
            r["types_present"] = sorted({e.get("type") for e in ents if e.get("type")})
            by = {}
            for e in ents:
                by.setdefault(e["type"], []).append(e["text"])
            r["entities_pretty"] = " | ".join(f"{t}: {', '.join(v)}" for t, v in sorted(by.items()))
            rows.append(r)
    M.assign_reasons(rows, qthresh)
    for r in rows:
        if r["filter_reason"] is None and en_context_tokens(r["text"]) < MIN_EN_CONTEXT:
            r["filter_reason"] = "entity_only_switch"
        elif r["filter_reason"] is None and not has_latin_entity_candidate(r["text"]):
            r["filter_reason"] = "arabic_script_entity"
        elif r["filter_reason"] is None and not r["entities_verified"]:
            r["filter_reason"] = "no_verified_entities"
    return rows


# --------------------------------------------------------------------------- merge
def cmd_merge(a):
    pool, seen = [], set()
    for p in sorted(DAILY.glob("run_*_filtered.jsonl")):
        for r in M._read_jsonl(p):
            k = M._norm(r.get("text", ""))
            if k and k not in seen:
                seen.add(k); pool.append(r)
    if not pool:
        raise SystemExit("empty pool — run `generate` first")
    by_group = {}
    for r in pool:
        by_group.setdefault(r["group"], []).append(r)

    target = a.target
    rng = random.Random(7)
    chosen, shortfall = [], {}
    for key, (must, _mn, _mx, share) in GROUP_KEY.items():
        want = round(share * target)
        have = by_group.get(key, [])
        rng.shuffle(have)
        chosen.extend(have[:want])
        if len(have) < want:
            shortfall[key] = want - len(have)
    # top up to exactly `target` from whatever remains, largest groups first
    if len(chosen) < target:
        picked = {M._norm(r["text"]) for r in chosen}
        rest = [r for r in pool if M._norm(r["text"]) not in picked]
        rng.shuffle(rest)
        chosen.extend(rest[: target - len(chosen)])
    chosen = chosen[:target]
    rng.shuffle(chosen)

    MERGED.mkdir(parents=True, exist_ok=True)
    base = MERGED / f"switchlingua_ner_train_{len(chosen)}"
    M._write_jsonl(base.with_suffix(".jsonl"), chosen)
    M._write_csv(base.with_suffix(".csv"), chosen)
    print(f"[merge] pool={len(pool)} -> selected {len(chosen)} (target {target})")
    print(f"[merge] by group: {dict(Counter(r['group'] for r in chosen))}")
    print(f"[merge] shortfall vs corpus proportions: {shortfall or 'none'}")
    print(f"[merge] wrote {base.with_suffix('.jsonl')}")


# --------------------------------------------------------------------------- bio
_TOK = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def to_bio(text, entities):
    """Character-offset -> token BIO tags. Mechanical; entity spans come from the validator."""
    toks = [(m.group(), m.start(), m.end()) for m in _TOK.finditer(text)]
    tags = ["O"] * len(toks)
    for e in sorted(entities, key=lambda x: -len(x["text"])):
        span, etype = e["text"], e["type"]
        for m in re.finditer(re.escape(span), text):
            s, t = m.start(), m.end()
            idx = [i for i, (_w, ts, te) in enumerate(toks) if ts >= s and te <= t]
            if not idx or any(tags[i] != "O" for i in idx):
                continue
            tags[idx[0]] = f"B-{etype}"
            for i in idx[1:]:
                tags[i] = f"I-{etype}"
            break
    return [w for w, _s, _e in toks], tags


def cmd_bio(a):
    src = MERGED / f"switchlingua_ner_train_{a.target}.jsonl"
    if not src.exists():
        raise SystemExit(f"{src} not found — run `merge --target {a.target}` first")
    rows = M._read_jsonl(src)
    out, tagged = [], 0
    for r in rows:
        toks, tags = to_bio(r["text"], r.get("entities_verified") or [])
        if any(t != "O" for t in tags):
            tagged += 1
        out.append({"tokens": toks, "ner_tags": tags, "text": r["text"],
                    "types_present": r.get("types_present", []), "source": "switchlingua_expN"})
    dst = MERGED / f"switchlingua_ner_train_{a.target}_bio.jsonl"
    M._write_jsonl(dst, out)
    lbl = Counter(t for r in out for t in r["ner_tags"] if t != "O")
    print(f"[bio] {len(out)} sentences | {tagged} carry >=1 entity ({100*tagged/len(out):.0f}%)")
    print(f"[bio] tag counts: {dict(lbl)}")
    print(f"[bio] wrote {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--max", type=int, default=120)
    g.add_argument("--concurrency", type=int, default=4); g.add_argument("--date", default=None)
    g.add_argument("--seed", type=int, default=7); g.add_argument("--config", default=None)
    g.add_argument("--quality-threshold", type=float, default=QTHRESH)
    g.add_argument("--groups", nargs="*", default=None,
                   help="restrict to entity-type groups still short of target, e.g. --groups PER LOC PER+LOC")
    g.set_defaults(func=cmd_generate)
    r = sub.add_parser("merge"); r.add_argument("--target", type=int, default=240); r.set_defaults(func=cmd_merge)
    b = sub.add_parser("bio"); b.add_argument("--target", type=int, default=240); b.set_defaults(func=cmd_bio)
    a = ap.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
