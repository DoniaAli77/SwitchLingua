"""Per-agent capture for sentiment agent-design variants B and D on the Ahmed
frozen-primary full_agentic@0.7 escalated subset (semantic_v1 prompts).

Deterministic (temp 0) re-run of each design's escalated samples, capturing the
lexical / logic / contextual / polarity slots so we can report per-agent accuracy
and inter-agent agreement (items 9-11). No training, no generation.

Run with no args; processes both B and D.
"""
from __future__ import annotations
import os, sys, json, glob, csv, collections, time

# OpenAI key from the shared .env
for line in open('../Modified_Version/.env', encoding='utf-8'):
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    k, v = line.split('=', 1)
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ['SENTIMENT_PROMPT_VARIANT'] = 'semantic_v1'

from src.config.loader import load_task_bundle
from src.state.schema import PipelineState, StateMetadata
from evaluate_pipeline import build_primary_classifier, build_llm_client, build_orchestrator

D = 'data/Sentiment/external/ahmed'
DESIGNS = [
    ("B", "polarity_contextual", "experiment_ahmed_designB_polarity_contextual",
     ["logic", "contextual"]),  # active agents (Polarity is in the logic slot; lexical abstains)
    ("D", "lexical_logic_contextual_polarity", "experiment_ahmed_designD_four_agent",
     ["lexical", "logic", "contextual", "polarity"]),
]


def ao(x):
    if x is None:
        return None
    return x.model_output.label


def cap_label(st, slot):
    return ao(getattr(st, f"{slot}_output"))


for tag, variant, outdir, active in DESIGNS:
    os.environ['SENTIMENT_AGENT_VARIANT'] = variant
    RUN = f'experiments/outputs/multi_agent_bert/{outdir}'
    OUT = RUN + '/error_attribution'
    os.makedirs(OUT, exist_ok=True)
    pr = json.load(open(glob.glob(RUN + '/full/*predictions.json')[0], encoding='utf-8'))
    esc = [x for x in pr if str(x.get('escalated')).lower() == 'true' or x.get('escalated') is True]
    print(f"\n##### Design {tag} ({variant}) — {len(esc)} escalated #####", flush=True)

    bundle = load_task_bundle('src/config/default.yaml', active_task='sentiment_classification',
                              pipeline_mode='full_agentic', threshold=0.7)
    tc = bundle.task_config
    primary = build_primary_classifier('precomputed', precomputed_predictions=D + '/ahmed_eesa_test_predictions_aligned.csv')
    llm = build_llm_client('openai', llm_model='gpt-4o-mini', allowed_labels=tc.labels)
    orch = build_orchestrator(tc, threshold=0.7, enable_deliberation=False,
                              keyword_map=bundle.keyword_map, rule_map=bundle.rule_map,
                              primary_classifier=primary, llm_client=llm,
                              consensus_primary_weight=1.0, sentiment_agent_variant=variant)
    rows = []
    for i, s in enumerate(esc):
        sid = s['sample_id']; text = s['input_text']; true = s['true_label']
        st = None
        for attempt in range(8):
            st = PipelineState(metadata=StateMetadata(sample_id=sid), input_text=text, task_config=tc)
            st = orch.run(st)
            need = active + ['__final__']
            ok = st.final_output and st.final_output.label is not None and all(
                cap_label(st, sl) is not None for sl in active)
            if ok:
                break
            time.sleep(15)
        pm = st.primary_model
        labels = {sl: cap_label(st, sl) for sl in ['lexical', 'logic', 'contextual', 'polarity']}
        f_lbl = st.final_output.label if st.final_output else None
        a_lbl = pm.label
        a_ok = a_lbl == true; f_ok = f_lbl == true
        trans = ('correct_to_correct' if a_ok and f_ok else 'correct_to_wrong' if a_ok and not f_ok
                 else 'wrong_to_correct' if not a_ok and f_ok else 'wrong_to_wrong')
        rows.append(dict(sample_id=sid, text=text.replace('\n', ' ')[:120], true_label=true,
                         ahmed_label=a_lbl, ahmed_conf=round(pm.confidence, 4),
                         lexical_label=labels['lexical'], logic_label=labels['logic'],
                         contextual_label=labels['contextual'], polarity_label=labels['polarity'],
                         final_label=f_lbl, ahmed_correct=a_ok, final_correct=f_ok, transition=trans))
        if (i + 1) % 10 == 0: print('  done', i + 1, '/', len(esc), flush=True)
        time.sleep(0.7)

    json.dump(rows, open(OUT + '/attribution_table.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    with open(OUT + '/attribution_table.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    n = len(rows)
    # map slot->display name for active agents
    disp = {"lexical": "lexical", "logic": ("polarity" if variant == "polarity_contextual" else "logic"),
            "contextual": "contextual", "polarity": "polarity"}

    def acc(slot):
        v = [r for r in rows if r[f'{slot}_label'] is not None]
        return (round(sum(1 for r in v if r[f'{slot}_label'] == r['true_label']) / len(v), 4), len(v))
    print(f"--- Design {tag} per-agent accuracy (active: {[disp[a] for a in active]}) ---")
    for a in active:
        ac, c = acc(a); print(f"   {disp[a]:12s} acc={ac:.4f} (n={c})")
    fac = round(sum(1 for r in rows if r['final_correct']) / n, 4)
    print(f"   {'final':12s} acc={fac:.4f}   ahmed=0.7500")
    # agreement among active agents
    def lab(r, slot): return r[f'{slot}_label']
    allagree = sum(1 for r in rows if len({lab(r, a) for a in active}) == 1 and all(lab(r, a) is not None for a in active))
    print(f"   all-active-agree {allagree}/{n} = {allagree/n:.4f}")
    for x in range(len(active)):
        for y in range(x + 1, len(active)):
            sx, sy = active[x], active[y]
            vv = [r for r in rows if lab(r, sx) and lab(r, sy)]
            ag = round(sum(1 for r in vv if lab(r, sx) == lab(r, sy)) / len(vv), 4)
            print(f"   {disp[sx]}<->{disp[sy]} = {ag:.4f}")
    print(f"   transitions: {dict(collections.Counter(r['transition'] for r in rows))}")
    print(f"   saved -> {OUT}/attribution_table.csv", flush=True)
