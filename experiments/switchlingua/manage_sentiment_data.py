"""
manage_sentiment_data.py — accumulation workflow for Experiment C sentiment data.
=================================================================================
Generation-management utilities AROUND the SwitchLingua sentiment generator. Does NOT change the
core pipeline, prompts, or (frozen) NER — it only orchestrates runs and post-processes their output.

Because the full 324-scenario run exceeds the daily request quota, this accumulates data safely
across multiple days:

  migrate  — one-time: package the existing 114-sentence pilot as pilot_v1/ (never overwritten)
             and seed the resume manifest from the first run's scenarios.
  generate — RESUME-aware run: skip scenarios already completed (stable scenario_id), APPEND new raw
             states to the day's file, update the manifest, then filter the day's data.
  merge    — combine pilot_v1 + every daily run into one de-duplicated, label-balanced dataset.

Filtering (every stage): non-empty text -> TaskValidator passed -> deterministic CS-valid ->
quality >= threshold -> duplicate removal (normalized text). Loss-by-reason is tracked.

Layout (under multi-agent-bert/data/Sentiment/generated/):
  pilot_v1/   raw_outputs.jsonl  filtered_train.csv/.jsonl  GENERATION_REPORT.md  stats.json
  daily_runs/ run_YYYYMMDD_raw.jsonl  run_YYYYMMDD_filtered.csv/.jsonl  run_YYYYMMDD_stats.json
  merged/     switchlingua_sentiment_train_merged.csv/.jsonl  MERGE_REPORT.md
  completed_scenarios.json   (resume manifest)

Usage:
  python experiments/switchlingua/manage_sentiment_data.py migrate
  python experiments/switchlingua/manage_sentiment_data.py generate --max 120 --concurrency 4
  python experiments/switchlingua/manage_sentiment_data.py merge --target-per-label 150
"""
import argparse, asyncio, csv, datetime, hashlib, importlib, json, os, pathlib, random, re, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
CONFIG = ROOT / "experiments" / "switchlingua" / "config_sentiment_expC.yaml"

GEN_DIR = ROOT / "multi-agent-bert" / "data" / "Sentiment" / "generated"
PILOT_DIR = GEN_DIR / "pilot_v1"
DAILY_DIR = GEN_DIR / "daily_runs"
MERGED_DIR = GEN_DIR / "merged"
MANIFEST = GEN_DIR / "completed_scenarios.json"
RAW_BACKUP = GEN_DIR / "_raw_pipeline" / "Arabic.jsonl"   # rich full-state backup from the first run
LEGACY_CSV = GEN_DIR / "switchlingua_sentiment_train_pilot.csv"
LEGACY_REPORT = ROOT / "multi-agent-bert" / "experiments" / "outputs" / "multi_agent_bert" / "generated_sentiment_data" / "GENERATION_REPORT.md"

QUALITY_THRESHOLD = 7.0
MODEL = "gpt-4o-mini"

CANON_COLS = [
    "text", "label", "topic", "cs_ratio", "cs_type", "cs_function", "tense", "perspective",
    "conversation_type", "intensity", "ambiguity", "scenario_id", "task_validator_passed",
    "cs_valid", "cs_ar_ratio", "cs_en_ratio", "fluency", "naturalness", "quality_score",
    "validator_predicted_label", "gender", "age", "education_level", "source",
]
REASONS = ["empty_text", "validator_failed", "not_cs_valid", "low_quality", "duplicate"]


# --------------------------------------------------------------------------- helpers
def _norm(t):
    return re.sub(r"\s+", " ", (t or "").strip()).lower()


def _scenario_id(d):
    """Stable id from the scenario's defining dimensions (works on a scenario OR a saved state)."""
    tc = d.get("task_constraints") or {}
    parts = [d.get("task"), d.get("label"), d.get("topic"), d.get("cs_ratio"), d.get("cs_type"),
             d.get("cs_function"), d.get("tense"), d.get("perspective"), d.get("conversation_type"),
             d.get("gender"), d.get("age"), d.get("education_level"),
             tc.get("intensity"), tc.get("ambiguity")]
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12]


def _num(d, *keys):
    if isinstance(d, dict):
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return round(float(d[k]), 3)
                except (TypeError, ValueError):
                    return None
    try:
        return round(float(d), 3)
    except (TypeError, ValueError):
        return None


_CS = None
def _cs_stats(text):
    global _CS
    if _CS is None:
        if str(MODIFIED_CORE) not in sys.path:
            sys.path.insert(0, str(MODIFIED_CORE))
        sys.modules.pop("utils", None)
        from utils import compute_true_cs_stats
        _CS = compute_true_cs_stats
    return _CS(text)


def extract_rows(state, source):
    """One row per sentence_record, with the full required metadata. No filtering applied here."""
    tc = state.get("task_constraints") or {}
    sid = _scenario_id(state)
    tvres = state.get("task_validation_results_per_instances") or []
    base = {
        "label": state.get("label"), "topic": state.get("topic"), "cs_ratio": state.get("cs_ratio"),
        "cs_type": state.get("cs_type"), "cs_function": state.get("cs_function"),
        "tense": state.get("tense"), "perspective": state.get("perspective"),
        "conversation_type": state.get("conversation_type"),
        "intensity": tc.get("intensity"), "ambiguity": tc.get("ambiguity"),
        "scenario_id": sid, "gender": state.get("gender"), "age": state.get("age"),
        "education_level": state.get("education_level"), "source": source,
    }
    rows = []
    for sr in state.get("sentence_records") or []:
        text = sr.get("text", "") or ""
        idx = sr.get("index")
        tvp = sr.get("task_passed")
        if tvp is None:
            tvp = (sr.get("task_validation") or {}).get("passed")
        st = _cs_stats(text) if text.strip() else {"is_code_switched": False, "cs_ar_ratio": 0.0, "cs_en_ratio": 0.0}
        pred = None
        if isinstance(idx, int) and idx < len(tvres):
            pred = (tvres[idx] or {}).get("predicted_label")
        rows.append({
            **base, "text": text,
            "task_validator_passed": bool(tvp),
            "cs_valid": bool(st["is_code_switched"]),
            "cs_ar_ratio": st.get("cs_ar_ratio"), "cs_en_ratio": st.get("cs_en_ratio"),
            "fluency": _num(sr.get("fluency"), "fluency_score"),
            "naturalness": _num(sr.get("naturalness"), "naturalness_score"),
            "quality_score": _num(sr.get("weighted_score")),
            "validator_predicted_label": pred,
        })
    return rows


def assign_reasons(rows, threshold):
    """Tag each row with filter_reason (None = kept); duplicate among otherwise-kept rows."""
    seen = set()
    for r in rows:
        if not (r["text"] or "").strip():
            r["filter_reason"] = "empty_text"
        elif not r["task_validator_passed"]:
            r["filter_reason"] = "validator_failed"
        elif not r["cs_valid"]:
            r["filter_reason"] = "not_cs_valid"
        elif r["quality_score"] is None or r["quality_score"] < threshold:
            r["filter_reason"] = "low_quality"
        else:
            k = _norm(r["text"])
            if k in seen:
                r["filter_reason"] = "duplicate"
            else:
                seen.add(k)
                r["filter_reason"] = None
    return rows


def funnel(rows):
    loss = Counter(r["filter_reason"] for r in rows if r["filter_reason"])
    kept = [r for r in rows if r["filter_reason"] is None]
    return {
        "raw_instances": len(rows),
        "raw_by_label": dict(Counter(r["label"] for r in rows)),
        "kept": len(kept),
        "kept_by_label": dict(Counter(r["label"] for r in kept)),
        "kept_by_topic": dict(Counter(r["topic"] for r in kept)),
        "loss_by_reason": {k: loss.get(k, 0) for k in REASONS},
        "duplicates_removed": loss.get("duplicate", 0),
    }, kept


def _read_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def _write_jsonl(p, rows, append=False):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a" if append else "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def _write_csv(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CANON_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CANON_COLS})


def _manifest_path(path=None):
    return pathlib.Path(path) if path else MANIFEST


def _load_manifest(path=None):
    p = _manifest_path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"total_scenarios": None, "completed": {}, "updated": None}


def _save_manifest(m, path=None):
    p = _manifest_path(path)
    m["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def _all_scenarios(config_path=None):
    """Build the full sentiment scenarios for a config + attach scenario_id (no API)."""
    if str(MODIFIED_CORE) not in sys.path:
        sys.path.insert(0, str(MODIFIED_CORE))
    sys.modules.pop("utils", None)
    import utils as ut
    cfg = ut.load_config(str(config_path or CONFIG))
    scn = [s for s in ut.generate_scenarios(cfg["pre_execute"]) if s.get("task") == "sentiment"]
    for s in scn:
        s["scenario_id"] = _scenario_id(s)
    return scn


# --------------------------------------------------------------------------- migrate
def cmd_migrate(args):
    if PILOT_DIR.exists() and (PILOT_DIR / "filtered_train.csv").exists() and not args.force:
        print(f"pilot_v1 already exists at {PILOT_DIR} (use --force to rebuild). Nothing overwritten.")
        return
    PILOT_DIR.mkdir(parents=True, exist_ok=True)

    states = [s for s in _read_jsonl(RAW_BACKUP) if s.get("task") == "sentiment"]
    if not states:
        raise SystemExit(f"no first-run states found at {RAW_BACKUP}")
    # pilot raw_outputs = full states, annotated with scenario_id + source
    for s in states:
        s["scenario_id"] = _scenario_id(s)
        s["source"] = "pilot_v1"
    _write_jsonl(PILOT_DIR / "raw_outputs.jsonl", states)

    # enrich the EXISTING 114 balanced selection with full metadata (join by normalized text)
    all_rows = []
    for s in states:
        all_rows.extend(extract_rows(s, "pilot_v1"))
    by_text = {_norm(r["text"]): r for r in all_rows}
    legacy = _read_jsonl(GEN_DIR / "switchlingua_sentiment_train_pilot.jsonl")
    enriched = []
    for old in legacy:
        full = by_text.get(_norm(old.get("text", "")))
        enriched.append(full if full else {**{c: old.get(c, "") for c in CANON_COLS}, "source": "pilot_v1"})
    _write_csv(PILOT_DIR / "filtered_train.csv", enriched)
    _write_jsonl(PILOT_DIR / "filtered_train.jsonl", enriched)

    # funnel over the pilot raw (informational); kept_by_label taken from the preserved 114 selection
    stats, _ = funnel(assign_reasons([dict(r) for r in all_rows], args.quality_threshold))
    stats.update({
        "source": "pilot_v1", "date": "first_run",
        "requested": len(states), "completed": len(states), "skipped_resumed": 0, "failed": None,
        "kept": len(enriched), "kept_by_label": dict(Counter(r["label"] for r in enriched)),
        "kept_by_topic": dict(Counter(r["topic"] for r in enriched)),
        "note": "kept counts reflect the preserved 114-sentence balanced pilot; funnel is informational.",
    })
    (PILOT_DIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # preserve the original GENERATION_REPORT.md inside pilot_v1
    if LEGACY_REPORT.exists():
        (PILOT_DIR / "GENERATION_REPORT.md").write_text(LEGACY_REPORT.read_text(encoding="utf-8"), encoding="utf-8")

    # seed resume manifest with the first run's scenario_ids
    m = _load_manifest()
    m["total_scenarios"] = len(_all_scenarios())
    for s in states:
        m["completed"][s["scenario_id"]] = {"source": "pilot_v1", "date": "first_run"}
    _save_manifest(m)

    print(f"[migrate] pilot_v1 built: {len(states)} raw states, {len(enriched)} balanced rows preserved.")
    print(f"[migrate] manifest seeded: {len(m['completed'])}/{m['total_scenarios']} scenarios completed.")
    print(f"[migrate] originals left untouched at {GEN_DIR} (nothing overwritten).")


# --------------------------------------------------------------------------- generate (resume + append)
def cmd_generate(args):
    date = args.date or datetime.date.today().strftime("%Y%m%d")
    cfg_path = getattr(args, "config", None)
    man_path = getattr(args, "manifest", None)
    raw_path = DAILY_DIR / f"run_{date}_raw.jsonl"
    scenarios = _all_scenarios(cfg_path)
    m = _load_manifest(man_path)
    m["total_scenarios"] = len(scenarios)
    done_ids = set(m["completed"].keys())
    todo = [s for s in scenarios if s["scenario_id"] not in done_ids]
    random.seed(args.seed); random.shuffle(todo)
    attempted = todo[: args.max] if args.max else todo
    print(f"[gen] config={pathlib.Path(cfg_path).name if cfg_path else 'default'} "
          f"manifest={_manifest_path(man_path).name} | total={len(scenarios)} completed={len(done_ids)} "
          f"remaining={len(todo)} attempting={len(attempted)} (date={date}, concurrency={args.concurrency})")
    if not attempted:
        print("[gen] nothing to do — all scenarios completed. Run `merge`.")
        return

    # env + SSL + core (only here, when actually calling the API)
    import dotenv
    env = ROOT / "Modified_Version" / ".env"
    if env.exists():
        dotenv.load_dotenv(str(env), override=True)
    os.environ["API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
    os.environ["API_BASE"] = os.environ.get("OPENAI_BASE_URL", "")
    import ssl, httpx
    ssl._create_default_https_context = ssl._create_unverified_context
    _c = httpx.Client.__init__
    httpx.Client.__init__ = lambda self, *a, **k: (k.setdefault("verify", False), _c(self, *a, **k))[-1]
    _a = httpx.AsyncClient.__init__
    httpx.AsyncClient.__init__ = lambda self, *a, **k: (k.setdefault("verify", False), _a(self, *a, **k))[-1]
    if str(MODIFIED_CORE) not in sys.path:
        sys.path.insert(0, str(MODIFIED_CORE))
    for mod in ("utils", "node_engine", "node_models", "prompt", "agents", "run_french"):
        sys.modules.pop(mod, None)
    importlib.invalidate_caches()
    import node_engine as ne, run_french as rf
    ne.MODEL = MODEL
    (GEN_DIR / "_raw_pipeline").mkdir(parents=True, exist_ok=True)
    ne.OUTPUT_DIR = str(GEN_DIR / "_raw_pipeline")  # agent's own backup; we persist returned states

    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrency)
    prog = {"done": 0, "ok": 0, "fail": 0}

    async def one(sc):
        async with sem:
            try:
                st = await rf.CodeSwitchingAgent(sc).run()
            except Exception as exc:
                async with lock:
                    prog["fail"] += 1
                    if "429" in str(exc) or "rate_limit" in str(exc).lower():
                        if prog["fail"] % 15 == 1:
                            print(f"[gen] rate-limited (429) — stopping productive work; remaining will skip. {exc}".split(" Visit")[0])
                    else:
                        print(f"[gen] SKIP {sc['scenario_id']}: {type(exc).__name__}: {str(exc)[:120]}")
                st = None
            async with lock:
                prog["done"] += 1
                if st:
                    st["scenario_id"] = sc["scenario_id"]
                    st["source"] = f"run_{date}"
                    _write_jsonl(raw_path, [st], append=True)
                    m["completed"][sc["scenario_id"]] = {"source": f"run_{date}", "date": date}
                    prog["ok"] += 1
                    if prog["ok"] % 20 == 0:
                        _save_manifest(m, man_path)
                if prog["done"] % 20 == 0 or prog["done"] == len(attempted):
                    print(f"[gen] {prog['done']}/{len(attempted)} ok={prog['ok']} fail={prog['fail']}", flush=True)

    asyncio.run(_gather(attempted, one))
    _save_manifest(m, man_path)

    # filter the day's accumulated raw -> daily filtered + stats
    states = [s for s in _read_jsonl(raw_path) if s.get("task") == "sentiment"]
    rows = []
    for s in states:
        rows.extend(extract_rows(s, f"run_{date}"))
    stats, kept = funnel(assign_reasons(rows, args.quality_threshold))
    _write_csv(DAILY_DIR / f"run_{date}_filtered.csv", kept)
    _write_jsonl(DAILY_DIR / f"run_{date}_filtered.jsonl", kept)
    # CS-valid rate by cs_ratio target (50% vs 60%) — drives the cs_ratio-prune stopping rule
    cs_by_ratio = {}
    nonempty = [r for r in rows if (r["text"] or "").strip()]
    for rr in sorted({r["cs_ratio"] for r in nonempty}):
        grp = [r for r in nonempty if r["cs_ratio"] == rr]
        v = sum(1 for r in grp if r["cs_valid"])
        cs_by_ratio[rr] = {"cs_valid": v, "total": len(grp), "pct": round(100 * v / len(grp), 1) if grp else None}
    pos_kept = stats["kept_by_label"].get("positive", 0)
    pos_per_scn = round(pos_kept / prog["ok"], 2) if prog["ok"] else None
    stats.update({
        "source": f"run_{date}", "date": date,
        "requested": len(todo), "attempted": len(attempted), "completed": prog["ok"],
        "skipped_resumed": len(done_ids), "failed": prog["fail"],
        "cs_valid_by_cs_ratio": cs_by_ratio, "positive_kept_per_scenario": pos_per_scn,
    })
    (DAILY_DIR / f"run_{date}_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gen] day {date}: ok={prog['ok']} fail={prog['fail']} | raw={stats['raw_instances']} kept={stats['kept']} "
          f"loss={stats['loss_by_reason']}")
    print(f"[gen] CS-valid by cs_ratio: " + " ".join(f"{k}={v['cs_valid']}/{v['total']}({v['pct']}%)" for k, v in cs_by_ratio.items()))
    print(f"[gen] positive kept/scenario = {pos_per_scn} (guardrail: pause if < 0.8)")
    print(f"[gen] manifest now {len(m['completed'])}/{m['total_scenarios']} completed. Then run `merge`.")


async def _gather(items, coro):
    await asyncio.gather(*(coro(x) for x in items))


# --------------------------------------------------------------------------- merge
def cmd_merge(args):
    sources = []
    pilot = _read_jsonl(PILOT_DIR / "filtered_train.jsonl")
    if pilot:
        sources.append(("pilot_v1", pilot))
    for p in sorted(DAILY_DIR.glob("run_*_filtered.jsonl")):
        sources.append((p.stem.replace("_filtered", ""), _read_jsonl(p)))
    if not sources:
        raise SystemExit("nothing to merge (run migrate / generate first).")

    # cross-source dedup by normalized text (pilot first, then daily in name order)
    seen, merged, cross_dup = set(), [], 0
    for src, rows in sources:
        for r in rows:
            r.setdefault("source", src)
            k = _norm(r.get("text", ""))
            if not k or k in seen:
                cross_dup += 1
                continue
            seen.add(k)
            merged.append(r)

    by_label = Counter(r["label"] for r in merged)
    by_topic = Counter(r["topic"] for r in merged)
    present = [l for l in ("positive", "negative", "neutral") if by_label.get(l, 0) > 0]
    cap = min(args.target_per_label, min(by_label[l] for l in present)) if present else 0
    rng = random.Random(7)
    balanced = []
    for l in present:
        rows = [r for r in merged if r["label"] == l]
        rng.shuffle(rows)
        balanced.extend(rows[:cap])
    rng.shuffle(balanced)
    bal_by_label = Counter(r["label"] for r in balanced)

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(MERGED_DIR / "switchlingua_sentiment_train_merged.csv", balanced)
    _write_jsonl(MERGED_DIR / "switchlingua_sentiment_train_merged.jsonl", balanced)

    # aggregate per-source stats for the report
    agg_raw = Counter(); agg_keptL = Counter(); agg_keptT = Counter(); agg_loss = Counter()
    per_src = []
    for sj in ([PILOT_DIR / "stats.json"] + sorted(DAILY_DIR.glob("run_*_stats.json"))):
        if sj.exists():
            d = json.loads(sj.read_text(encoding="utf-8"))
            per_src.append(d)
            agg_raw.update({k: v for k, v in (d.get("raw_by_label") or {}).items()})
            agg_keptL.update({k: v for k, v in (d.get("kept_by_label") or {}).items()})
            agg_keptT.update({k: v for k, v in (d.get("kept_by_topic") or {}).items()})
            agg_loss.update({k: v for k, v in (d.get("loss_by_reason") or {}).items()})

    m = _load_manifest(getattr(args, "manifest", None))
    total = m.get("total_scenarios") or len(_all_scenarios(getattr(args, "config", None)))
    completed = len(m.get("completed", {}))
    remaining_scn = max(0, total - completed)
    kept_per_scn = (len(merged) / completed) if completed else 0
    min_label = min((by_label[l] for l in present), default=0)
    need_per_label = max(0, args.target_per_label - min_label)
    est_examples_needed = need_per_label * 3
    est_scn_needed = round(est_examples_needed / kept_per_scn) if kept_per_scn else None

    _write_merge_report(sources, merged, by_label, by_topic, balanced, bal_by_label, cap, cross_dup,
                        per_src, agg_raw, agg_keptL, agg_keptT, agg_loss, total, completed, remaining_scn,
                        kept_per_scn, args.target_per_label, min_label, est_examples_needed, est_scn_needed)

    print(f"[merge] sources={len(sources)} merged_unique={len(merged)} cross_dups={cross_dup}")
    print(f"[merge] by_label={dict(by_label)}  balanced({cap}/label)={dict(bal_by_label)} TOTAL={sum(bal_by_label.values())}")
    print(f"[merge] completed {completed}/{total} scenarios | target {args.target_per_label}/label -> "
          f"need ~{est_examples_needed} more kept (~{est_scn_needed} more scenarios)")
    print(f"[merge] outputs -> {MERGED_DIR}")


def _write_merge_report(sources, merged, by_label, by_topic, balanced, bal_by_label, cap, cross_dup,
                        per_src, agg_raw, agg_keptL, agg_keptT, agg_loss, total, completed, remaining_scn,
                        kept_per_scn, target, min_label, est_examples_needed, est_scn_needed):
    L = ["# Experiment C — Sentiment Data MERGE Report\n",
         f"Combined **{len(sources)}** source(s) into one de-duplicated, label-balanced sentiment dataset. "
         "Pilot_v1 + daily runs; NER frozen/untouched; pipeline & prompts unchanged.\n",
         "## Scenario coverage (resume manifest)",
         f"- requested (full design): **{total}** scenarios",
         f"- completed so far: **{completed}**",
         f"- remaining: **{remaining_scn}**",
         f"- observed yield: **{kept_per_scn:.2f}** kept examples / completed scenario\n",
         "## Sources merged",
         "| source | rows in | ", "|---|--:|"]
    for src, rows in sources:
        L.append(f"| {src} | {len(rows)} |")
    L += [f"\n## Merged dataset (pre-balance, cross-source de-duplicated)",
          f"- unique examples: **{len(merged)}**  (cross-source duplicates removed: **{cross_dup}**)",
          "\n**By label:**", "| label | count |", "|---|--:|"]
    for l in ("positive", "negative", "neutral"):
        L.append(f"| {l} | {by_label.get(l, 0)} |")
    L += ["\n**By topic:**", "| topic | count |", "|---|--:|"]
    for t, c in sorted(by_topic.items(), key=lambda x: -x[1]):
        L.append(f"| {t} | {c} |")
    L += [f"\n## Balanced training set (down-sampled to {cap}/label)",
          "| label | count |", "|---|--:|"]
    for l in ("positive", "negative", "neutral"):
        L.append(f"| {l} | {bal_by_label.get(l, 0)} |")
    L += [f"| **TOTAL** | **{sum(bal_by_label.values())}** |\n",
          "## Cumulative funnel (aggregated across source stats.json)",
          f"- raw by label: {dict(agg_raw)}",
          f"- kept by label: {dict(agg_keptL)}",
          "- filtering loss by reason:",
          "", "| reason | count |", "|---|--:|"]
    for r in REASONS:
        L.append(f"| {r} | {agg_loss.get(r, 0)} |")
    L += ["\n## Kept by topic (cumulative)", "| topic | kept |", "|---|--:|"]
    for t, c in sorted(agg_keptT.items(), key=lambda x: -x[1]):
        L.append(f"| {t} | {c} |")
    L += [f"\n## Estimated remaining work (target = {target}/label)",
          f"- current smallest label: **{min_label}** (need **{max(0, target - min_label)}** more/label)",
          f"- estimated additional KEPT examples needed: **~{est_examples_needed}**",
          f"- estimated additional SCENARIOS to run: **~{est_scn_needed}** (at current yield; "
          f"{remaining_scn} scenarios remain in the 324-design — run across days under the daily quota)\n",
          "## Outputs",
          "- `merged/switchlingua_sentiment_train_merged.csv` / `.jsonl` — balanced, ready for Experiment C.",
          "- per-row metadata: " + ", ".join(CANON_COLS) + ".\n",
          "## Notes",
          "- Multi-Agent BERT is **NOT trained here** — accumulation only.",
          "- Add more data: `manage_sentiment_data.py generate --max <N>` on a new day, then `merge` again.",
          "- `on_execute.round` / `shared.style` remain inert; size grows by running more scenarios."]
    (MERGED_DIR / "MERGE_REPORT.md").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    mg = sub.add_parser("migrate", help="package existing 114 pilot as pilot_v1 + seed manifest")
    mg.add_argument("--quality-threshold", type=float, default=QUALITY_THRESHOLD)
    mg.add_argument("--force", action="store_true", help="rebuild pilot_v1 even if it exists")
    mg.set_defaults(func=cmd_migrate)

    gn = sub.add_parser("generate", help="resume-aware generation, append + filter the day's run")
    gn.add_argument("--max", type=int, default=120, help="cap scenarios this run (daily quota safety; 0 = all remaining)")
    gn.add_argument("--concurrency", type=int, default=4)
    gn.add_argument("--date", default=None, help="YYYYMMDD label (default: today)")
    gn.add_argument("--seed", type=int, default=7)
    gn.add_argument("--quality-threshold", type=float, default=QUALITY_THRESHOLD)
    gn.add_argument("--config", default=None, help="scenario config (default: config_sentiment_expC.yaml)")
    gn.add_argument("--manifest", default=None, help="resume manifest path (default: completed_scenarios.json)")
    gn.set_defaults(func=cmd_generate)

    mr = sub.add_parser("merge", help="combine pilot_v1 + daily runs into one balanced dataset")
    mr.add_argument("--target-per-label", type=int, default=150, help="balance target per label")
    mr.add_argument("--config", default=None, help="scenario config for the remaining-scenario estimate")
    mr.add_argument("--manifest", default=None, help="resume manifest for the coverage estimate")
    mr.set_defaults(func=cmd_merge)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
