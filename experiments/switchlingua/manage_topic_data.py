"""
manage_topic_data.py — accumulation workflow for the 9-class TOPIC training dataset (Exp T).
============================================================================================
Same resume-safe pattern as manage_sentiment_data.py (whose helpers it reuses), but:
  * task = "topic" (9 labels), config config_topic_expT_v1.yaml
  * OWN isolated tree: multi-agent-bert/data/Topic/generated/  (never touches the Sentiment pool)
  * label-DYNAMIC balancing (no hardcoded positive/negative/neutral)
  * first dataset generated under the task-aware pipeline defaults (TASK_AWARE_ACCEPT=1,
    task-aware meet_criteria) — noted in the stats for provenance.

Filters (unchanged): non-empty -> TaskValidator passed -> deterministic CS-valid ->
quality >= 7.0 -> dedup (normalized text).

Usage:
  python experiments/switchlingua/manage_topic_data.py generate --max 30 --concurrency 4
  python experiments/switchlingua/manage_topic_data.py merge --target-per-label 50
"""
import argparse, asyncio, datetime, importlib, json, os, pathlib, random, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
SW = ROOT / "experiments" / "switchlingua"
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
sys.path.insert(0, str(SW))

import manage_sentiment_data as M  # shared helpers: extract_rows/assign_reasons/funnel/io/_scenario_id

CONFIG = SW / "config_topic_expT_v1.yaml"
TGEN = ROOT / "multi-agent-bert" / "data" / "Topic" / "generated"
DAILY = TGEN / "daily_runs"
MERGED = TGEN / "merged"
MANIFEST = TGEN / "completed_scenarios_topic.json"
MODEL = "gpt-4o-mini"
QTHRESH = 7.0


def _load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"total_scenarios": None, "completed": {}, "updated": None}


def _save_manifest(m):
    m["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def _all_scenarios(config_path=None):
    if str(MODIFIED_CORE) not in sys.path:
        sys.path.insert(0, str(MODIFIED_CORE))
    sys.modules.pop("utils", None)
    import utils as ut
    cfg = ut.load_config(str(config_path or CONFIG))
    scn = [s for s in ut.generate_scenarios(cfg["pre_execute"]) if s.get("task") == "topic"]
    for s in scn:
        s["scenario_id"] = M._scenario_id(s)
    return scn


# --------------------------------------------------------------------------- generate
def cmd_generate(args):
    date = args.date or datetime.date.today().strftime("%Y%m%d")
    raw_path = DAILY / f"run_{date}_raw.jsonl"
    scenarios = _all_scenarios(args.config)
    m = _load_manifest()
    m["total_scenarios"] = len(scenarios)
    done = set(m["completed"].keys())
    todo = [s for s in scenarios if s["scenario_id"] not in done]
    random.seed(args.seed); random.shuffle(todo)
    attempted = todo[: args.max] if args.max else todo
    print(f"[topic-gen] total={len(scenarios)} completed={len(done)} remaining={len(todo)} "
          f"attempting={len(attempted)} (date={date}, TASK_AWARE_ACCEPT env={os.getenv('TASK_AWARE_ACCEPT','<default 1>')})")
    if not attempted:
        print("[topic-gen] nothing to do — run `merge`.")
        return

    # env + SSL + core (mirrors the sentiment workflow)
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
    (TGEN / "_backup").mkdir(parents=True, exist_ok=True)
    ne.OUTPUT_DIR = str(TGEN / "_backup")

    DAILY.mkdir(parents=True, exist_ok=True)
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrency)
    prog = {"d": 0, "ok": 0, "f": 0}

    async def one(sc):
        async with sem:
            try:
                st = await rf.CodeSwitchingAgent(sc).run()
            except Exception as exc:
                st = None
                async with lock:
                    prog["f"] += 1
                    msg = str(exc)
                    if ("429" in msg or "rate_limit" in msg.lower()) and prog["f"] % 15 == 1:
                        print("[topic-gen] 429 rate-limited — remaining will skip")
                    elif "429" not in msg:
                        print(f"[topic-gen] SKIP {sc['scenario_id']}: {type(exc).__name__}: {msg[:100]}")
            async with lock:
                prog["d"] += 1
                if st:
                    st["scenario_id"] = sc["scenario_id"]
                    st["source"] = f"run_{date}"
                    M._write_jsonl(raw_path, [st], append=True)
                    m["completed"][sc["scenario_id"]] = {"source": f"run_{date}", "date": date}
                    prog["ok"] += 1
                    if prog["ok"] % 20 == 0:
                        _save_manifest(m)
                if prog["d"] % 20 == 0 or prog["d"] == len(attempted):
                    print(f"[topic-gen] {prog['d']}/{len(attempted)} ok={prog['ok']} fail={prog['f']}", flush=True)

    asyncio.run(_gather(attempted, one))
    _save_manifest(m)

    # filter the day's raw -> daily filtered + stats (labels dynamic)
    states = [s for s in M._read_jsonl(raw_path) if s.get("task") == "topic"]
    rows = []
    for s in states:
        rows.extend(M.extract_rows(s, f"run_{date}"))
    M.assign_reasons(rows, args.quality_threshold)
    kept = [r for r in rows if r["filter_reason"] is None]
    M._write_csv(DAILY / f"run_{date}_filtered.csv", kept)
    M._write_jsonl(DAILY / f"run_{date}_filtered.jsonl", kept)
    nonempty = [r for r in rows if (r["text"] or "").strip()]
    stats = {
        "task": "topic", "source": f"run_{date}", "date": date,
        "task_aware_accept": os.getenv("TASK_AWARE_ACCEPT", "1"),
        "requested": len(todo), "attempted": len(attempted), "completed": prog["ok"], "failed": prog["f"],
        "raw_instances": len(rows),
        "cs_valid_rate_pct": round(100 * sum(1 for r in nonempty if r["cs_valid"]) / len(nonempty), 1) if nonempty else 0,
        "kept": len(kept),
        "kept_by_label": dict(Counter(r["label"] for r in kept)),
        "loss_by_reason": dict(Counter(r["filter_reason"] for r in rows if r["filter_reason"])),
    }
    (DAILY / f"run_{date}_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[topic-gen] day {date}: ok={prog['ok']} fail={prog['f']} | raw={stats['raw_instances']} "
          f"kept={stats['kept']} cs_valid={stats['cs_valid_rate_pct']}% loss={stats['loss_by_reason']}")
    print(f"[topic-gen] kept by label: {stats['kept_by_label']}")
    ks = stats["kept_by_label"]
    if prog["ok"]:
        print(f"[topic-gen] kept/scenario = {stats['kept']/prog['ok']:.2f} | manifest {len(m['completed'])}/{len(scenarios)}")


async def _gather(items, coro):
    await asyncio.gather(*(coro(x) for x in items))


# --------------------------------------------------------------------------- merge
def cmd_merge(args):
    sources = [(p.stem.replace("_filtered", ""), M._read_jsonl(p)) for p in sorted(DAILY.glob("run_*_filtered.jsonl"))]
    if not sources:
        raise SystemExit("nothing to merge — run generate first.")
    seen, pool, dup = set(), [], 0
    for src, rows in sources:
        for r in rows:
            r.setdefault("source", src)
            k = M._norm(r.get("text", ""))
            if not k or k in seen:
                dup += 1
                continue
            seen.add(k)
            pool.append(r)

    by_label = Counter(r["label"] for r in pool)
    labels = sorted(by_label)
    cap = min(args.target_per_label, min(by_label.values())) if by_label else 0
    rng = random.Random(7)
    balanced = []
    for lab in labels:
        rows = [r for r in pool if r["label"] == lab]
        rng.shuffle(rows)
        balanced.extend(rows[:cap])
    rng.shuffle(balanced)

    MERGED.mkdir(parents=True, exist_ok=True)
    total = len(balanced)
    base = MERGED / (args.out_name or f"switchlingua_topic_train_{total}_{cap}perlabel")
    M._write_csv(base.with_suffix(".csv"), balanced)
    M._write_jsonl(base.with_suffix(".jsonl"), balanced)

    print(f"[topic-merge] sources={len(sources)} pool={len(pool)} cross_dups={dup}")
    print(f"[topic-merge] pool by label: {dict(by_label)}")
    print(f"[topic-merge] balanced({cap}/label x {len(labels)} labels) = {total} -> {base.with_suffix('.csv')}")
    need = {lab: max(0, args.target_per_label - by_label[lab]) for lab in labels if by_label[lab] < args.target_per_label}
    print(f"[topic-merge] short of target {args.target_per_label}/label: {need or 'NONE — target reached'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="resume-aware topic generation + filter the day's run")
    g.add_argument("--max", type=int, default=50)
    g.add_argument("--concurrency", type=int, default=4)
    g.add_argument("--date", default=None)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--quality-threshold", type=float, default=QTHRESH)
    g.add_argument("--config", default=None)
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("merge", help="combine daily runs into one balanced 9-class dataset")
    r.add_argument("--target-per-label", type=int, default=50)
    r.add_argument("--out-name", default=None)
    r.set_defaults(func=cmd_merge)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
