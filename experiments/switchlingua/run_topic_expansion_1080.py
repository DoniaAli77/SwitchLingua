"""
run_topic_expansion_1080.py — 540 additional TOPIC sentences in 3 independent batches
=====================================================================================
Generates 3 x 180 (20 per topic per batch) NEW sentences under the UNCHANGED original
config (`config_topic_expT_v1.yaml`, cs_ratio ["50%","60%"]), then combines them with the
existing Topic-540 into a balanced Topic-1080 (120 per topic).

CONFIGURATION IS NOT MODIFIED. The baseline manifest is already 288/288, so the new
sentences come from RE-SAMPLING the same 288 scenario specifications (the generator is
stochastic). Scenario ids are namespaced ":p2" in this run's OWN manifest so the baseline
manifest and pool are never touched or re-used.

ISOLATION: everything lands in data/Topic/generated/expansion_1080/. The baseline
daily_runs/, manifest and merged corpora are opened READ-ONLY (as a dedup blocklist).

PER BATCH (all four required steps):
  1. validate every generation constraint - the unmodified filter chain from
     manage_sentiment_data: non-empty -> TaskValidator passed -> deterministic CS-valid
     -> quality >= 7.0, then dedup;
  2. deduplicate within the batch, against Topic-540, and against all previous batches;
  3. save the accepted batch IMMEDIATELY (batch{n}_accepted.jsonl + batch{n}_report.json);
  4. quota-safe: every completed scenario is recorded in the manifest before the next
     wave, so a 429 stop resumes at the next unfinished batch and never regenerates
     sentences that were already accepted.

No model training is performed by this script.

Usage:
  python experiments/switchlingua/run_topic_expansion_1080.py run          # all batches
  python experiments/switchlingua/run_topic_expansion_1080.py run --batch 2
  python experiments/switchlingua/run_topic_expansion_1080.py combine      # build Topic-1080
  python experiments/switchlingua/run_topic_expansion_1080.py status
"""
import argparse, asyncio, datetime, importlib, json, math, os, pathlib, random, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
SW = ROOT / "experiments" / "switchlingua"
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
sys.path.insert(0, str(SW))

import manage_sentiment_data as M
from profile_topic_corpus import agg

CONFIG = SW / "config_topic_expT_v1.yaml"          # UNCHANGED original config
TGEN = ROOT / "multi-agent-bert" / "data" / "Topic" / "generated"
BASE = TGEN / "expansion_1080"
RAWD = BASE / "raw"
ACCD = BASE / "batches"
MANIFEST = BASE / "completed_scenarios_expansion.json"
TOPIC540 = TGEN / "merged" / "switchlingua_topic_train_540_60perlabel.jsonl"
OUT1080 = TGEN / "merged" / "switchlingua_topic_train_1080_120perlabel"

MODEL = "gpt-4o-mini"
QTHRESH = 7.0
NAMESPACE = "p2"
PER_LABEL = 20
BATCHES = [1, 2, 3]
LABELS = ["business", "education", "health", "shopping", "medical",
          "sports", "tech", "finance", "social"]


# --------------------------------------------------------------------------- helpers
def _manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"completed": {}}


def _save_manifest(m):
    m["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def all_scenarios(passes=range(2, 8)):
    """The same 288 scenario specs, offered as several independent RE-SAMPLING passes.

    The baseline run consumed pass 1 (288/288 in the baseline manifest). Each extra pass
    re-runs the identical specs with the stochastic generator and gets its own id suffix
    (":p2", ":p3", ...), so no scenario is ever run twice within this study and the three
    batches cannot exhaust each other. The CONFIG itself is untouched."""
    if str(MODIFIED_CORE) not in sys.path:
        sys.path.insert(0, str(MODIFIED_CORE))
    sys.modules.pop("utils", None)
    import utils as ut
    cfg = ut.load_config(str(CONFIG))
    scn = [s for s in ut.generate_scenarios(cfg["pre_execute"]) if s.get("task") == "topic"]
    out = []
    for p in passes:
        for s in scn:
            c = dict(s)
            c["scenario_id"] = M._scenario_id(s) + f":p{p}"
            out.append(c)
    return out


def topic540_keys():
    if not TOPIC540.exists():
        raise SystemExit(f"baseline corpus missing: {TOPIC540}")
    return {M._norm(r.get("text", "")) for r in M._read_jsonl(TOPIC540)}


def prior_batch_rows(upto_batch):
    rows = []
    for b in BATCHES:
        if b >= upto_batch:
            continue
        p = ACCD / f"batch{b}_accepted.jsonl"
        if p.exists():
            rows.extend(M._read_jsonl(p))
    return rows


def filter_batch(batch, blocklist):
    """Full unmodified filter chain, then cross-corpus dedup. Idempotent: always
    recomputed from the batch's raw file, so a resumed run is identical to a clean one."""
    raw = RAWD / f"batch{batch}_raw.jsonl"
    states = [s for s in M._read_jsonl(raw) if s.get("task") == "topic"]
    rows = []
    for s in states:
        rows.extend(M.extract_rows(s, s.get("source", f"expansion_b{batch}")))
    M.assign_reasons(rows, QTHRESH)           # empty / validator / cs_valid / quality / dup
    seen = set()
    for r in rows:
        if r["filter_reason"] is not None:
            continue
        k = M._norm(r["text"])
        if k in blocklist:
            r["filter_reason"] = "dup_vs_existing"    # Topic-540 or an earlier batch
        elif k in seen:
            r["filter_reason"] = "duplicate"          # within this batch
        else:
            seen.add(k)
    accepted = [r for r in rows if r["filter_reason"] is None]
    return rows, accepted


def batch_report(batch, rows, accepted, selected, n_scen):
    nonempty = [r for r in rows if (r["text"] or "").strip()]
    n_ne = len(nonempty) or 1
    prof = agg(selected) if selected else {}
    rep = {
        "batch": batch,
        "scenarios_completed_total": n_scen,
        "candidate_sentences": len(rows),
        "accepted_available": len(accepted),
        "selected_for_batch": len(selected),
        "per_label": dict(Counter(r["label"] for r in selected)),
        "cs_valid_rate_pct": round(100 * sum(1 for r in nonempty if r["cs_valid"]) / n_ne, 1),
        "loss_by_reason": dict(Counter(r["filter_reason"] for r in rows if r["filter_reason"])),
        "quality_min": min((r["quality_score"] for r in selected), default=None),
        "ar_pct": prof.get("ar_pct"), "cmi_mean": prof.get("cmi_mean"),
        "switch_mean": prof.get("switch_mean"), "len_mean": prof.get("len_mean"),
        "cs_valid_pct_selected": prof.get("cs_valid_pct"),
        "duplicates_in_selected": prof.get("dups"),
        "saved_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
    }
    (ACCD / f"batch{batch}_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return rep


# --------------------------------------------------------------------------- generation
def boot():
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
    if str(MODIFIED_CORE) not in sys.path:
        sys.path.insert(0, str(MODIFIED_CORE))
    for mod in ("utils", "node_engine", "node_models", "prompt", "agents", "run_french"):
        sys.modules.pop(mod, None)
    importlib.invalidate_caches()
    import node_engine as ne, run_french as rf
    ne.MODEL = MODEL
    (BASE / "_backup").mkdir(parents=True, exist_ok=True)
    ne.OUTPUT_DIR = str(BASE / "_backup")
    print(f"[exp] model={MODEL} config={CONFIG.name} TASK_AWARE_ACCEPT="
          f"{os.getenv('TASK_AWARE_ACCEPT', '<default 1>')}")
    return ne, rf


async def _wave(rf, scens, concurrency, batch, man):
    raw = RAWD / f"batch{batch}_raw.jsonl"
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    prog = {"d": 0, "ok": 0, "f": 0, "q": 0}

    async def one(sc):
        async with sem:
            try:
                st = await rf.CodeSwitchingAgent(dict(sc)).run()
            except Exception as exc:
                st = None
                msg = str(exc)
                async with lock:
                    prog["f"] += 1
                    if "429" in msg or "rate_limit" in msg.lower() or "quota" in msg.lower():
                        prog["q"] += 1
                        if prog["q"] % 15 == 1:
                            print("[exp] 429 / quota pressure")
                    else:
                        print(f"[exp] SKIP {sc['scenario_id']}: {type(exc).__name__}: {msg[:110]}")
            async with lock:
                prog["d"] += 1
                if st:
                    st["scenario_id"] = sc["scenario_id"]
                    st["source"] = f"expansion_b{batch}"
                    M._write_jsonl(raw, [st], append=True)
                    man["completed"][sc["scenario_id"]] = {"batch": batch}
                    prog["ok"] += 1
                    if prog["ok"] % 10 == 0:
                        _save_manifest(man)
                if prog["d"] % 10 == 0 or prog["d"] == len(scens):
                    print(f"[exp] b{batch} {prog['d']}/{len(scens)} ok={prog['ok']} "
                          f"fail={prog['f']} (429={prog['q']})", flush=True)

    await asyncio.gather(*(one(s) for s in scens))
    _save_manifest(man)
    return prog


def run_batch(batch, a, scen_by_label, man, ne, rf):
    """Fill one batch to exactly 20 accepted per label. Returns 'done' | 'quota' | 'exhausted'."""
    RAWD.mkdir(parents=True, exist_ok=True)
    ACCD.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed + batch)
    block = topic540_keys() | {M._norm(r["text"]) for r in prior_batch_rows(batch)}
    print(f"\n[exp] ===== BATCH {batch} ===== blocklist={len(block)} texts")

    for wave in range(1, a.max_waves + 1):
        rows, accepted = filter_batch(batch, block)
        have = Counter(r["label"] for r in accepted)
        need = {l: PER_LABEL - have.get(l, 0) for l in LABELS if have.get(l, 0) < PER_LABEL}
        if not need:
            break
        done_ids = set(man["completed"])
        done_b = sum(1 for i in done_ids if man["completed"][i].get("batch") == batch)
        # Seed with the measured baseline yield (629 kept / 288 scenarios = 2.18) until this
        # batch has enough of its own evidence; otherwise wave 1 would over-order massively.
        y = (len(accepted) / done_b) if done_b >= 10 else 2.18
        y = max(0.5, y)
        ask, avail_total = {}, 0
        for lab, n in need.items():
            avail = [s for s in scen_by_label[lab] if s["scenario_id"] not in done_ids]
            avail_total += len(avail)
            ask[lab] = min(len(avail), math.ceil(n / y) + 2, a.max_per_label_wave)
        if not sum(ask.values()):
            print(f"[exp] b{batch} scenario space exhausted, still need {need}")
            return "exhausted"
        pick = []
        for lab, k in ask.items():
            av = [s for s in scen_by_label[lab] if s["scenario_id"] not in done_ids]
            rng.shuffle(av)
            pick.extend(av[:k])
        rng.shuffle(pick)
        print(f"[exp] b{batch} wave {wave}: need {need} | yield~{y:.2f} | "
              f"running {len(pick)} scenarios")
        prog = asyncio.run(_wave(rf, pick, a.concurrency, batch, man))
        if prog["ok"] == 0 and prog["q"] > 0:
            print(f"[exp] b{batch} QUOTA: every scenario in the wave failed with 429.")
            return "quota"

    # select exactly PER_LABEL per label, save immediately
    rows, accepted = filter_batch(batch, block)
    have = Counter(r["label"] for r in accepted)
    short = {l: PER_LABEL - have.get(l, 0) for l in LABELS if have.get(l, 0) < PER_LABEL}
    if short:
        print(f"[exp] b{batch} INCOMPLETE, short {short} — nothing saved for this batch. "
              f"Raw states and the manifest are preserved; re-running resumes here.")
        return "incomplete"
    rng2 = random.Random(7 + batch)
    selected = []
    for lab in LABELS:
        pool = [r for r in accepted if r["label"] == lab]
        rng2.shuffle(pool)
        selected.extend(pool[:PER_LABEL])
    rng2.shuffle(selected)
    M._write_jsonl(ACCD / f"batch{batch}_accepted.jsonl", selected)
    M._write_csv(ACCD / f"batch{batch}_accepted.csv", selected)
    n_scen = sum(1 for i in man["completed"] if man["completed"][i].get("batch") == batch)
    rep = batch_report(batch, rows, accepted, selected, n_scen)
    print(f"[exp] BATCH {batch} SAVED: {len(selected)} sentences "
          f"({PER_LABEL}/label), scenarios={n_scen}, "
          f"cs_valid={rep['cs_valid_rate_pct']}%, loss={rep['loss_by_reason']}")
    return "done"


def cmd_run(a):
    scen = all_scenarios()
    by_label = {l: [s for s in scen if s["label"] == l] for l in LABELS}
    man = _manifest()
    print(f"[exp] scenario space={len(scen)} (32/label), manifest={len(man['completed'])} used")
    ne, rf = boot()
    todo = [a.batch] if a.batch else BATCHES
    for b in todo:
        if (ACCD / f"batch{b}_accepted.jsonl").exists():
            print(f"[exp] batch {b} already complete — skipping (no regeneration)")
            continue
        status = run_batch(b, a, by_label, man, ne, rf)
        if status != "done":
            print(f"\n[exp] STOPPED SAFELY at batch {b} (status={status}). "
                  f"Re-run the same command to resume; completed scenarios are in the manifest "
                  f"and finished batches are never regenerated.")
            return 3
    print("\n[exp] all three batches complete — run `combine` to build Topic-1080.")
    return 0


# --------------------------------------------------------------------------- combine
def cmd_combine(a):
    missing = [b for b in BATCHES if not (ACCD / f"batch{b}_accepted.jsonl").exists()]
    if missing:
        raise SystemExit(f"batches not finished: {missing}")
    base = M._read_jsonl(TOPIC540)
    new = []
    for b in BATCHES:
        new.extend(M._read_jsonl(ACCD / f"batch{b}_accepted.jsonl"))
    combined = base + new
    keys = [M._norm(r["text"]) for r in combined]
    dups = len(keys) - len(set(keys))
    per = Counter(r["label"] for r in combined)
    if dups or len(combined) != 1080 or set(per.values()) != {120}:
        raise SystemExit(f"integrity failure: n={len(combined)} dups={dups} per_label={dict(per)}")
    random.Random(11).shuffle(combined)
    M._write_jsonl(OUT1080.with_suffix(".jsonl"), combined)
    M._write_csv(OUT1080.with_suffix(".csv"), combined)
    prof_all, prof_new = agg(combined), agg(new)
    summary = {
        "total": len(combined), "per_label": dict(per), "duplicates": dups,
        "existing_540": len(base), "new_540": len(new),
        "profile_1080": {k: prof_all[k] for k in
                         ("len_mean", "ar_pct", "en_pct", "cmi_mean", "switch_mean", "cs_valid_pct")},
        "profile_new540": {k: prof_new[k] for k in
                           ("len_mean", "ar_pct", "en_pct", "cmi_mean", "switch_mean", "cs_valid_pct")},
        "batch_reports": [json.loads((ACCD / f"batch{b}_report.json").read_text(encoding="utf-8"))
                          for b in BATCHES],
    }
    (BASE / "TOPIC_1080_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[exp] Topic-1080 written: {OUT1080.with_suffix('.jsonl')}")
    print(f"[exp] {len(combined)} sentences, 120/label, {dups} duplicates")
    print(f"[exp] new-540 profile: {summary['profile_new540']}")
    print(f"[exp] 1080 profile   : {summary['profile_1080']}")
    return 0


def cmd_status(a):
    man = _manifest()
    print(f"manifest scenarios used: {len(man['completed'])}")
    for b in BATCHES:
        p = ACCD / f"batch{b}_accepted.jsonl"
        if p.exists():
            rows = M._read_jsonl(p)
            print(f"  batch {b}: COMPLETE  {len(rows)} sentences "
                  f"{dict(Counter(r['label'] for r in rows))}")
        else:
            raw = RAWD / f"batch{b}_raw.jsonl"
            print(f"  batch {b}: pending   (raw states: {len(M._read_jsonl(raw)) if raw.exists() else 0})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--batch", type=int, default=None, choices=BATCHES)
    r.add_argument("--max-waves", type=int, default=8)
    r.add_argument("--max-per-label-wave", type=int, default=16,
                   help="cap scenarios ordered per label per wave (stops over-ordering)")
    r.add_argument("--seed", type=int, default=7)
    r.set_defaults(func=cmd_run)
    c = sub.add_parser("combine"); c.set_defaults(func=cmd_combine)
    s = sub.add_parser("status"); s.set_defaults(func=cmd_status)
    a = ap.parse_args()
    sys.exit(a.func(a))


if __name__ == "__main__":
    main()
