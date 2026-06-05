"""
run_expC_sentiment_generation.py — Multi-Agent BERT Experiment C: sentiment data generation.
=============================================================================================
Drives the Modified_Version SwitchLingua pipeline (gpt-4o-mini) over a SENTIMENT-ONLY config
to produce balanced Arabic-English code-switched sentiment training data, then FILTERS:
  - keep only TaskValidator-passed instances
  - keep only deterministic CS-valid instances (compute_true_cs_stats.is_code_switched)
  - keep only quality >= threshold (per-sentence weighted_score; default 7.0 = calibrated bar)
  - de-duplicate (normalized text)
  - balance labels (down-sample to the smallest kept label)
  - report kept count per label

NOTE: NER is frozen and untouched (this run is sentiment-only). on_execute.round / shared.style
are inert in the pipeline (see config header); size is set by the config Cartesian product.

Generation phase calls the live API (long). Filtering re-runs offline from the raw jsonl.

Usage:
  python experiments/switchlingua/run_expC_sentiment_generation.py            # generate + filter
  python experiments/switchlingua/run_expC_sentiment_generation.py --filter-only   # re-filter raw
  python experiments/switchlingua/run_expC_sentiment_generation.py --count 30 --concurrency 4
Outputs:
  multi-agent-bert/data/Sentiment/generated/switchlingua_sentiment_train_pilot.{csv,jsonl}  (BALANCED)
  multi-agent-bert/data/Sentiment/generated/switchlingua_sentiment_kept_all.jsonl           (pre-balance)
  multi-agent-bert/data/Sentiment/generated/_raw_pipeline/raw_states.jsonl                  (raw pipeline)
  multi-agent-bert/experiments/outputs/multi_agent_bert/generated_sentiment_data/GENERATION_REPORT.md
"""
import argparse, asyncio, csv, importlib, json, os, pathlib, random, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
CONFIG = ROOT / "experiments" / "switchlingua" / "config_sentiment_expC.yaml"

GEN_DIR = ROOT / "multi-agent-bert" / "data" / "Sentiment" / "generated"
RAW_DIR = GEN_DIR / "_raw_pipeline"
RAW_JSONL = RAW_DIR / "raw_states.jsonl"
REPORT_DIR = ROOT / "multi-agent-bert" / "experiments" / "outputs" / "multi_agent_bert" / "generated_sentiment_data"

QUALITY_THRESHOLD = 7.0
MODEL = "gpt-4o-mini"

# ---- env + SSL (mirror run_full_pipeline_generation) ----
import dotenv as _dotenv
_env = ROOT / "Modified_Version" / ".env"
if _env.exists():
    _dotenv.load_dotenv(str(_env), override=True)
os.environ["API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
os.environ["API_BASE"] = os.environ.get("OPENAI_BASE_URL", "")
import ssl as _ssl, httpx as _httpx
_ssl._create_default_https_context = _ssl._create_unverified_context
_co = _httpx.Client.__init__
_httpx.Client.__init__ = lambda self, *a, **k: (k.setdefault("verify", False), _co(self, *a, **k))[-1]
_ao = _httpx.AsyncClient.__init__
_httpx.AsyncClient.__init__ = lambda self, *a, **k: (k.setdefault("verify", False), _ao(self, *a, **k))[-1]


def _norm(t):
    return re.sub(r"\s+", " ", (t or "").strip()).lower()


# ============================ GENERATION ============================
async def generate(count, concurrency):
    for d in (BASELINE := MODIFIED_CORE,):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    for m in ("utils", "node_engine", "node_models", "prompt", "agents", "run_french"):
        sys.modules.pop(m, None)
    importlib.invalidate_caches()
    import utils as ut, node_engine as ne, run_french as rf

    ne.MODEL = MODEL
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ne.OUTPUT_DIR = str(RAW_DIR)  # AcceptanceAgent backup file; we persist returned states ourselves

    cfg = ut.load_config(str(CONFIG))
    scenarios = [s for s in ut.generate_scenarios(cfg["pre_execute"]) if s.get("task") == "sentiment"]
    random.seed(7); random.shuffle(scenarios)
    if count:
        scenarios = scenarios[:count]
    print(f"[gen] {len(scenarios)} sentiment scenarios | model={MODEL} | concurrency={concurrency}")

    sem = asyncio.Semaphore(concurrency)
    done = {"n": 0, "fail": 0}
    states = []

    async def one(i, sc):
        async with sem:
            try:
                st = await rf.CodeSwitchingAgent(sc).run()
                if st:
                    states.append(st)
            except Exception as exc:
                done["fail"] += 1
                print(f"[gen] SKIP {i}: {type(exc).__name__}: {exc}")
            finally:
                done["n"] += 1
                if done["n"] % 20 == 0 or done["n"] == len(scenarios):
                    print(f"[gen] {done['n']}/{len(scenarios)} (fail={done['fail']})", flush=True)

    await asyncio.gather(*(one(i, sc) for i, sc in enumerate(scenarios)))

    with open(RAW_JSONL, "w", encoding="utf-8") as f:
        for st in states:
            rec = {k: st.get(k) for k in ("task", "label", "task_constraints", "topic", "cs_type",
                                          "data_generation_result", "sentence_records",
                                          "task_validation_results_per_instances")}
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"[gen] wrote {len(states)} raw scenario states -> {RAW_JSONL} (fail={done['fail']})")
    return len(scenarios), done["fail"]


# ============================ FILTERING ============================
def filter_and_write(threshold):
    if str(MODIFIED_CORE) not in sys.path:
        sys.path.insert(0, str(MODIFIED_CORE))
    sys.modules.pop("utils", None)
    from utils import compute_true_cs_stats

    if not RAW_JSONL.exists():
        raise SystemExit(f"raw states not found: {RAW_JSONL} (run generation first)")
    raw = [json.loads(l) for l in RAW_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

    funnel = {"raw_instances": 0, "validator_pass": 0, "cs_valid": 0, "quality_ok": 0, "after_dedup": 0}
    kept, seen = [], set()
    for rec in raw:
        if rec.get("task") != "sentiment":
            continue
        label = rec.get("label")
        tc = rec.get("task_constraints") or {}
        intensity, ambiguity = tc.get("intensity"), tc.get("ambiguity")
        topic, cs_type = rec.get("topic"), rec.get("cs_type")
        srs = rec.get("sentence_records") or []
        tvres = rec.get("task_validation_results_per_instances") or []
        for sr in srs:
            idx = sr.get("index")
            text = sr.get("text", "")
            if not text.strip():
                continue
            funnel["raw_instances"] += 1
            tv_passed = sr.get("task_passed")
            if tv_passed is None:
                tv_passed = (sr.get("task_validation") or {}).get("passed")
            if not tv_passed:
                continue
            funnel["validator_pass"] += 1
            st = compute_true_cs_stats(text)
            if not st["is_code_switched"]:
                continue
            funnel["cs_valid"] += 1
            q = sr.get("weighted_score")
            if q is None or float(q) < threshold:
                continue
            funnel["quality_ok"] += 1
            key = _norm(text)
            if key in seen:
                continue
            seen.add(key)
            pred = None
            if isinstance(idx, int) and idx < len(tvres):
                pred = (tvres[idx] or {}).get("predicted_label")
            kept.append({
                "text": text, "label": label, "intensity": intensity, "ambiguity": ambiguity,
                "topic": topic, "cs_type": cs_type, "quality_score": round(float(q), 3),
                "validator_predicted_label": pred,
                "cs_ar_ratio": st["cs_ar_ratio"], "cs_en_ratio": st["cs_en_ratio"],
            })
    funnel["after_dedup"] = len(kept)

    # kept-per-label (pre-balance)
    from collections import Counter
    kept_by_label = Counter(k["label"] for k in kept)

    # balance: down-sample to smallest kept label
    labels = ["positive", "negative", "neutral"]
    present = [l for l in labels if kept_by_label.get(l, 0) > 0]
    min_n = min((kept_by_label[l] for l in present), default=0)
    rng = random.Random(7)
    balanced = []
    for l in present:
        rows = [k for k in kept if k["label"] == l]
        rng.shuffle(rows)
        balanced.extend(rows[:min_n])
    rng.shuffle(balanced)
    bal_by_label = Counter(b["label"] for b in balanced)

    # ---- write outputs ----
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["text", "label", "intensity", "ambiguity", "topic", "cs_type",
            "quality_score", "validator_predicted_label", "cs_ar_ratio", "cs_en_ratio"]
    with open(GEN_DIR / "switchlingua_sentiment_train_pilot.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(balanced)
    with open(GEN_DIR / "switchlingua_sentiment_train_pilot.jsonl", "w", encoding="utf-8") as f:
        for r in balanced:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(GEN_DIR / "switchlingua_sentiment_kept_all.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return funnel, dict(kept_by_label), dict(bal_by_label), min_n, balanced, kept


def write_report(threshold, gen_meta, funnel, kept_by_label, bal_by_label, min_n, balanced, kept):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    n_scn, n_fail = gen_meta
    ex = balanced[:9]
    L = [
        "# Experiment C — SwitchLingua Sentiment Generation (pilot)\n",
        f"Generated Arabic-English code-switched **sentiment** training data with the Modified_Version "
        f"pipeline (`{MODEL}`), then filtered for trainable, balanced data. **NER untouched (frozen); "
        f"sentiment-only run.**\n",
        "## Generation",
        f"- Config: `experiments/switchlingua/config_sentiment_expC.yaml` (324 sentiment scenarios by design).",
        f"- Scenarios run: **{n_scn}** (failed: {n_fail}).",
        f"- Quality threshold (per-sentence weighted_score): **{threshold}**.\n",
        "## Filter funnel (instances)",
        "| stage | count |", "|---|--:|",
        f"| raw generated instances | {funnel['raw_instances']} |",
        f"| TaskValidator passed | {funnel['validator_pass']} |",
        f"| + deterministic CS-valid | {funnel['cs_valid']} |",
        f"| + quality >= {threshold} | {funnel['quality_ok']} |",
        f"| + de-duplicated (KEPT) | {funnel['after_dedup']} |\n",
        "## Kept per label (pre-balance)",
        "| label | kept |", "|---|--:|",
    ]
    for l in ("positive", "negative", "neutral"):
        L.append(f"| {l} | {kept_by_label.get(l, 0)} |")
    L += [f"\n## Balanced training set (down-sampled to smallest label = {min_n}/label)",
          "| label | count |", "|---|--:|"]
    for l in ("positive", "negative", "neutral"):
        L.append(f"| {l} | {bal_by_label.get(l, 0)} |")
    L += [f"| **TOTAL** | **{sum(bal_by_label.values())}** |\n",
          "## Outputs",
          "- `data/Sentiment/generated/switchlingua_sentiment_train_pilot.csv` / `.jsonl` — **balanced** training pilot.",
          "- `data/Sentiment/generated/switchlingua_sentiment_kept_all.jsonl` — all kept (pre-balance).",
          "- `data/Sentiment/generated/_raw_pipeline/raw_states.jsonl` — raw pipeline output.\n",
          "## Examples (balanced set)"]
    for e in ex:
        L.append(f"- [{e['label']}/{e['intensity']}] (q={e['quality_score']}, ar%={e['cs_ar_ratio']}) {e['text'][:90]}")
    L += ["\n## Notes / caveats",
          "- **Neutral as factual/descriptive:** the generation prompt is frozen, so neutral-specific wording "
          "could not be injected via config (only `intensity`/`ambiguity` flow into sentiment task_constraints). "
          "`ambiguity: low` was used for cleaner labels; neutral quality is enforced post-hoc by the TaskValidator "
          "filter. Residual risk of mildly-polar 'neutral' remains — recommend a human spot-check before training.",
          "- `on_execute.round` and `shared.style` are inert in the pipeline (not read by any code); dataset size "
          "is governed by the config Cartesian product, not those fields.",
          "- Filtering is reproducible offline: `run_expC_sentiment_generation.py --filter-only`.",
          "- **Multi-Agent BERT is NOT trained here** (data generation only).\n"]
    (REPORT_DIR / "GENERATION_REPORT.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=0, help="cap scenarios (0 = all 324)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--quality-threshold", type=float, default=QUALITY_THRESHOLD)
    ap.add_argument("--filter-only", action="store_true", help="skip generation; re-filter raw_states.jsonl")
    args = ap.parse_args()

    gen_meta = (0, 0)
    if not args.filter_only:
        gen_meta = asyncio.run(generate(args.count, args.concurrency))
    else:
        # recover scenario count from raw for the report
        if RAW_JSONL.exists():
            gen_meta = (sum(1 for l in RAW_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()), 0)

    funnel, kept_by_label, bal_by_label, min_n, balanced, kept = filter_and_write(args.quality_threshold)
    write_report(args.quality_threshold, gen_meta, funnel, kept_by_label, bal_by_label, min_n, balanced, kept)

    print(f"\n[filter] raw_instances={funnel['raw_instances']} -> validator={funnel['validator_pass']} "
          f"-> cs_valid={funnel['cs_valid']} -> quality>={args.quality_threshold}:{funnel['quality_ok']} "
          f"-> dedup(KEPT)={funnel['after_dedup']}")
    print(f"[filter] kept per label: {kept_by_label}")
    print(f"[filter] balanced ({min_n}/label): {bal_by_label}  TOTAL={sum(bal_by_label.values())}")
    print(f"[filter] outputs -> {GEN_DIR}")
    print(f"[report] -> {REPORT_DIR / 'GENERATION_REPORT.md'}")


if __name__ == "__main__":
    main()
