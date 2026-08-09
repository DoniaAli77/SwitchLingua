"""
run_topic_arabicdominant.py — ArabicDominant-180 (Experiment T-AD)
==================================================================
Builds a SEPARATE 180-sentence topic corpus (20 accepted per label x 9 labels) under
config_topic_expT_v2_arabicdominant.yaml, whose ONLY difference from the baseline
config is cs_ratio ["50%","60%"] -> ["85%","95%"].

Everything else is the frozen baseline: same prompts, same model (gpt-4o-mini), same
scenario dimensions, same quality pipeline, same filter chain, same acceptance
threshold (quality >= 7.0). The filter chain is not re-implemented here — it calls
manage_sentiment_data.extract_rows / assign_reasons, exactly as manage_topic_data does.

ISOLATION: everything is written under data/Topic/generated/variants/ArabicDominant/.
The baseline pool, its manifest and its merged corpora are never read for writing and
never appended to.

WAVE / CHECKPOINT PROTOCOL (as specified):
  * Wave 1 = 45 scenarios, exactly 5 per label. It is a CHECKPOINT, not a disposable
    pilot: every sentence it accepts counts toward the final 180.
  * After wave 1 the gate metrics are reported and tested:
        retained yield >= 0.9 accepted sentences per completed scenario
        Arabic tokens  >= 63.0 %
        mean CMI       <= 34.0
    Gate metrics are computed over the ACCEPTED sentences (that is the corpus being
    built), using the same tokenizer/CMI/switch-point definitions as the published
    TOPIC-540 profile (imported from profile_topic_corpus).
  * PASS  -> continue automatically, in per-label waves, until every label has 20.
    FAIL  -> stop. The configuration is NOT adjusted mid-run.

DEDUPLICATION: internal (normalized text, across all waves) and against the existing
TOPIC-540 corpus, which is loaded read-only as a blocklist.

No Silver corpus vocabulary, labels, predictions, errors or frequent terms are used
anywhere in generation. The only Silver-derived quantities are the numeric gate
thresholds, applied after the fact.

Usage:
  python experiments/switchlingua/run_topic_arabicdominant.py run --concurrency 4
  python experiments/switchlingua/run_topic_arabicdominant.py merge
  python experiments/switchlingua/run_topic_arabicdominant.py status
"""
import argparse, asyncio, datetime, importlib, json, math, os, pathlib, random, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
SW = ROOT / "experiments" / "switchlingua"
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
sys.path.insert(0, str(SW))

import manage_sentiment_data as M          # shared, unmodified filter/IO helpers
from profile_topic_corpus import agg       # same tokenizer / CMI / switch-point defs

CONFIG = SW / "config_topic_expT_v2_arabicdominant.yaml"
BASE = ROOT / "multi-agent-bert" / "data" / "Topic" / "generated" / "variants" / "ArabicDominant"
DAILY = BASE / "daily_runs"
MERGED = BASE / "merged"
MANIFEST = BASE / "completed_scenarios_ad.json"
CHECKPOINT_JSON = BASE / "checkpoint_45.json"
RAW = DAILY / "raw_all.jsonl"              # single cumulative raw file (append-only)

TOPIC540 = (ROOT / "multi-agent-bert" / "data" / "Topic" / "generated" / "merged"
            / "switchlingua_topic_train_540_60perlabel.jsonl")

MODEL = "gpt-4o-mini"
QTHRESH = 7.0
TARGET_PER_LABEL = 20
CHECKPOINT_PER_LABEL = 5
GATE_YIELD, GATE_AR, GATE_CMI = 0.9, 63.0, 34.0
LABELS = ["business", "education", "health", "shopping", "medical",
          "sports", "tech", "finance", "social"]


# --------------------------------------------------------------------------- scenarios
def all_scenarios():
    if str(MODIFIED_CORE) not in sys.path:
        sys.path.insert(0, str(MODIFIED_CORE))
    sys.modules.pop("utils", None)
    import utils as ut
    cfg = ut.load_config(str(CONFIG))
    scn = [s for s in ut.generate_scenarios(cfg["pre_execute"]) if s.get("task") == "topic"]
    for s in scn:
        s["scenario_id"] = M._scenario_id(s)
    return scn


def _manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"completed": {}}


def _save_manifest(m):
    m["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def _topic540_keys():
    """Read-only blocklist of the baseline corpus (normalized text)."""
    if not TOPIC540.exists():
        raise SystemExit(f"baseline corpus missing: {TOPIC540}")
    return {M._norm(r.get("text", "")) for r in M._read_jsonl(TOPIC540)}


# --------------------------------------------------------------------------- filtering
def rebuild(topic540_keys):
    """Re-filter the ENTIRE accumulated raw file from scratch (idempotent).

    Baseline chain via assign_reasons: empty -> validator_failed -> not_cs_valid ->
    low_quality -> duplicate. Two extra dedup reasons are layered on top:
    dup_vs_topic540 (cross-corpus) and duplicate (cross-wave, same normalized text).
    """
    states = [s for s in M._read_jsonl(RAW) if s.get("task") == "topic"]
    rows = []
    for s in states:
        rows.extend(M.extract_rows(s, s.get("source", "ad")))
    M.assign_reasons(rows, QTHRESH)
    seen = set()
    for r in rows:
        if r["filter_reason"] is not None:
            continue
        k = M._norm(r["text"])
        if k in topic540_keys:
            r["filter_reason"] = "dup_vs_topic540"
        elif k in seen:
            r["filter_reason"] = "duplicate"
        else:
            seen.add(k)
    accepted = [r for r in rows if r["filter_reason"] is None]
    return states, rows, accepted


def report(rows, accepted, n_scen, tag):
    """Gate metrics. Profile stats are over ACCEPTED sentences (the corpus being built)."""
    nonempty = [r for r in rows if (r["text"] or "").strip()]
    n_ne = len(nonempty) or 1
    cs_valid = sum(1 for r in nonempty if r["cs_valid"])
    mono_ar = sum(1 for r in nonempty
                  if not r["cs_valid"] and (r.get("cs_ar_ratio") or 0) > 0
                  and (r.get("cs_en_ratio") or 0) == 0)
    p = agg(accepted) if accepted else {}
    s = {
        "tag": tag,
        "scenarios_completed": n_scen,
        "candidate_sentences": len(rows),
        "candidate_yield_per_scenario": round(len(rows) / n_scen, 3) if n_scen else 0,
        "accepted": len(accepted),
        "retained_yield_per_scenario": round(len(accepted) / n_scen, 3) if n_scen else 0,
        "cs_valid_rate_pct": round(100 * cs_valid / n_ne, 1),
        "monolingual_arabic_rate_pct": round(100 * mono_ar / n_ne, 1),
        "monolingual_arabic_share_of_cs_failures_pct": round(
            100 * mono_ar / max(1, n_ne - cs_valid), 1),
        "ar_pct": p.get("ar_pct"), "en_pct": p.get("en_pct"),
        "cmi_mean": p.get("cmi_mean"), "cmi_median": p.get("cmi_median"),
        "switch_mean": p.get("switch_mean"), "len_mean": p.get("len_mean"),
        "accepted_by_label": dict(Counter(r["label"] for r in accepted)),
        "loss_by_reason": dict(Counter(r["filter_reason"] for r in rows if r["filter_reason"])),
    }
    print(f"\n===== {tag} =====")
    for k, v in s.items():
        if k != "tag":
            print(f"  {k:45} {v}")
    return s


def gates(s):
    checks = {
        "retained_yield >= 0.9": (s["retained_yield_per_scenario"], GATE_YIELD,
                                  s["retained_yield_per_scenario"] >= GATE_YIELD),
        "ar_pct >= 63.0": (s["ar_pct"], GATE_AR, (s["ar_pct"] or 0) >= GATE_AR),
        "cmi_mean <= 34.0": (s["cmi_mean"], GATE_CMI, (s["cmi_mean"] or 999) <= GATE_CMI),
    }
    print("\n----- GATES -----")
    for name, (got, want, ok) in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name:28} got={got}")
    return all(ok for _, _, ok in checks.values())


# --------------------------------------------------------------------------- generation
def boot():
    """Identical env/SSL/core bootstrap to manage_topic_data.cmd_generate."""
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
    print(f"[ad] gen model={MODEL} TASK_AWARE_ACCEPT={os.getenv('TASK_AWARE_ACCEPT', '<default 1>')}")
    return ne, rf


async def _wave(rf, scens, concurrency, date, man):
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    prog = {"d": 0, "ok": 0, "f": 0}

    async def one(sc):
        async with sem:
            try:
                st = await rf.CodeSwitchingAgent(dict(sc)).run()
            except Exception as exc:
                st = None
                async with lock:
                    prog["f"] += 1
                    msg = str(exc)
                    if ("429" in msg or "rate_limit" in msg.lower()):
                        if prog["f"] % 15 == 1:
                            print("[ad] 429 rate-limited")
                    else:
                        print(f"[ad] SKIP {sc['scenario_id']}: {type(exc).__name__}: {msg[:120]}")
            async with lock:
                prog["d"] += 1
                if st:
                    st["scenario_id"] = sc["scenario_id"]
                    st["source"] = f"run_{date}"
                    M._write_jsonl(RAW, [st], append=True)
                    man["completed"][sc["scenario_id"]] = {"date": date, "label": sc.get("label")}
                    prog["ok"] += 1
                    if prog["ok"] % 10 == 0:
                        _save_manifest(man)
                if prog["d"] % 10 == 0 or prog["d"] == len(scens):
                    print(f"[ad] {prog['d']}/{len(scens)} ok={prog['ok']} fail={prog['f']}", flush=True)

    await asyncio.gather(*(one(s) for s in scens))
    _save_manifest(man)
    return prog


def pick(scen_by_label, done_ids, per_label, rng):
    """`per_label` may be an int or a {label: n} mapping. Never re-picks a completed id."""
    out = []
    for lab in LABELS:
        n = per_label.get(lab, 0) if isinstance(per_label, dict) else per_label
        if n <= 0:
            continue
        avail = [s for s in scen_by_label[lab] if s["scenario_id"] not in done_ids]
        rng.shuffle(avail)
        out.extend(avail[:n])
    rng.shuffle(out)
    return out


def cmd_run(a):
    DAILY.mkdir(parents=True, exist_ok=True)
    date = datetime.date.today().strftime("%Y%m%d")
    rng = random.Random(a.seed)
    t540 = _topic540_keys()
    scen = all_scenarios()
    by_label = {lab: [s for s in scen if s["label"] == lab] for lab in LABELS}
    print(f"[ad] scenario space={len(scen)} ({ {l: len(v) for l, v in by_label.items()} })")
    print(f"[ad] TOPIC-540 blocklist: {len(t540)} normalized texts")

    man = _manifest()
    done = set(man["completed"])
    ne, rf = boot()

    # ---- wave 1: the checkpoint (5 per label = 45). Its output COUNTS. ----
    if len(done) == 0:
        w1 = pick(by_label, done, a.checkpoint_per_label, rng)
        print(f"[ad] CHECKPOINT WAVE: {len(w1)} scenarios ({a.checkpoint_per_label}/label)")
        asyncio.run(_wave(rf, w1, a.concurrency, date, man))
        done = set(man["completed"])
        _, rows, accepted = rebuild(t540)
        s = report(rows, accepted, len(done), f"CHECKPOINT ({len(done)} scenarios)")
        ok = gates(s)
        s["gates_passed"] = ok
        CHECKPOINT_JSON.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
        if not ok:
            print("\n[ad] CHECKPOINT FAILED -> stopping. Configuration NOT adjusted. "
                  "Accepted sentences are preserved in the raw file.")
            return 2
        print("\n[ad] CHECKPOINT PASSED -> continuing automatically toward 20/label.")
    else:
        print(f"[ad] resuming: {len(done)} scenarios already completed")
        _, rows, accepted = rebuild(t540)
        report(rows, accepted, len(done), "RESUME STATE")

    # ---- subsequent waves: only labels still short ----
    for wave in range(1, a.max_waves + 1):
        _, rows, accepted = rebuild(t540)
        have = Counter(r["label"] for r in accepted)
        need = {lab: TARGET_PER_LABEL - have.get(lab, 0) for lab in LABELS}
        need = {lab: n for lab, n in need.items() if n > 0}
        if not need:
            print("\n[ad] TARGET REACHED: 20 accepted for every label.")
            break
        y = max(0.4, len(accepted) / max(1, len(set(man['completed']))))
        ask = {}
        for lab, n in need.items():
            avail = sum(1 for s in by_label[lab] if s["scenario_id"] not in set(man["completed"]))
            ask[lab] = min(avail, math.ceil(n / y) + 2)
        if not sum(ask.values()):
            print(f"\n[ad] STOPPING: scenario space exhausted, still short {need}")
            break
        print(f"\n[ad] WAVE {wave}: still need {need} | yield={y:.2f} | asking {ask} "
              f"({sum(ask.values())} scenarios)")
        w = pick(by_label, set(man["completed"]), ask, rng)
        asyncio.run(_wave(rf, w, a.concurrency, date, man))

    _, rows, accepted = rebuild(t540)
    s = report(rows, accepted, len(set(man["completed"])), "FINAL")
    (BASE / "final_stats.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- merge
def cmd_merge(a):
    t540 = _topic540_keys()
    _, rows, accepted = rebuild(t540)
    have = Counter(r["label"] for r in accepted)
    short = {lab: TARGET_PER_LABEL - have.get(lab, 0) for lab in LABELS if have.get(lab, 0) < TARGET_PER_LABEL}
    if short:
        raise SystemExit(f"[ad-merge] cannot build exactly {TARGET_PER_LABEL}/label — short: {short}")
    rng = random.Random(7)
    out = []
    for lab in LABELS:
        pool = [r for r in accepted if r["label"] == lab]
        rng.shuffle(pool)
        out.extend(pool[:TARGET_PER_LABEL])
    rng.shuffle(out)
    MERGED.mkdir(parents=True, exist_ok=True)
    base = MERGED / f"switchlingua_topic_arabicdominant_{len(out)}_{TARGET_PER_LABEL}perlabel"
    M._write_jsonl(base.with_suffix(".jsonl"), out)
    M._write_csv(base.with_suffix(".csv"), out)
    print(f"[ad-merge] wrote {len(out)} ({TARGET_PER_LABEL}/label) -> {base.with_suffix('.jsonl')}")
    print(f"[ad-merge] by label: {dict(Counter(r['label'] for r in out))}")
    print(f"[ad-merge] overlap with TOPIC-540: "
          f"{sum(1 for r in out if M._norm(r['text']) in t540)} (must be 0)")
    return 0


def cmd_status(a):
    t540 = _topic540_keys()
    _, rows, accepted = rebuild(t540)
    report(rows, accepted, len(set(_manifest()["completed"])), "STATUS")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--checkpoint-per-label", type=int, default=CHECKPOINT_PER_LABEL)
    r.add_argument("--max-waves", type=int, default=6)
    r.add_argument("--seed", type=int, default=7)
    r.set_defaults(func=cmd_run)
    m_ = sub.add_parser("merge"); m_.set_defaults(func=cmd_merge)
    s_ = sub.add_parser("status"); s_.set_defaults(func=cmd_status)
    a = ap.parse_args()
    sys.exit(a.func(a))


if __name__ == "__main__":
    main()
