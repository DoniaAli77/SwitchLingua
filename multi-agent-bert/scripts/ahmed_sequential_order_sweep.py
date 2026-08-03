"""Sequential agent-ORDER sweep (Design C, Ahmed, gpt-4o-mini).

Holds everything fixed (same 3 agents, same prompts, same primary/threshold/LLM,
same consensus, sequential_chain=True) and varies ONLY the execution order of the
three voters via ``task_config.agent_stage_order``. Tests whether the order in
which chained agents see each other changes the outcome. No training/generation.
"""
from __future__ import annotations
import os, sys, json, csv, time, io

sys.path.insert(0, os.path.abspath('.'))
for line in open('../Modified_Version/.env', encoding='utf-8'):
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    k, v = line.split('=', 1)
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ['SENTIMENT_PROMPT_VARIANT'] = 'semantic_v1'
os.environ['SENTIMENT_AGENT_VARIANT'] = 'lexical_polarity_contextual'  # Design C

from sklearn.metrics import f1_score, accuracy_score
from src.config.loader import load_task_bundle
from src.state.schema import PipelineState, StateMetadata
from evaluate_pipeline import build_primary_classifier, build_llm_client, build_orchestrator

MODEL = 'gpt-4o-mini'; THRESH = 0.70
D = 'data/Sentiment/external/ahmed'
ALIGNED = D + '/ahmed_eesa_test_predictions_aligned.csv'
OUTDIR = 'experiments/outputs/multi_agent_bert/experiment_ahmed_sequential_order_sweep'
os.makedirs(OUTDIR, exist_ok=True)
ACTIVE = ['lexical', 'logic', 'contextual']
rows = list(csv.DictReader(open(ALIGNED, encoding='utf-8')))

# (display name, stage order). "logic" stage == the Polarity agent (variant C).
ORDERS = [
    ("L->P->C (default)", ["lexical_agent", "logic_agent", "contextual_agent"]),
    ("C->P->L (reverse)", ["contextual_agent", "logic_agent", "lexical_agent"]),
    ("P->L->C",           ["logic_agent", "lexical_agent", "contextual_agent"]),
    ("C->L->P",           ["contextual_agent", "lexical_agent", "logic_agent"]),
]


def run_order(order):
    bundle = load_task_bundle('src/config/default.yaml', active_task='sentiment_classification',
                              pipeline_mode='full_agentic', threshold=THRESH)
    tc = bundle.task_config
    tc.sequential_chain = True
    tc.agent_stage_order = order
    primary = build_primary_classifier('precomputed', precomputed_predictions=ALIGNED)
    llm = build_llm_client('openai', llm_model=MODEL, allowed_labels=tc.labels)
    orch = build_orchestrator(tc, threshold=THRESH, enable_deliberation=False,
                              keyword_map=bundle.keyword_map, rule_map=bundle.rule_map,
                              primary_classifier=primary, llm_client=llm,
                              consensus_primary_weight=1.0,
                              sentiment_agent_variant='lexical_polarity_contextual')
    recs = []; esc = 0; fallbacks = 0
    for i, r in enumerate(rows):
        sid = r['sample_id']; true = r['true_label']; text = r['text']
        escalated = float(r['confidence']) < THRESH
        pred = None
        for attempt in range(10):  # resilient to transient connection errors
            try:
                st = PipelineState(metadata=StateMetadata(sample_id=sid), input_text=text, task_config=tc)
                st = orch.run(st)
            except Exception as exc:  # network/API blip → back off and retry
                time.sleep(min(5 + attempt * 5, 45))
                continue
            ok = (st.final_output and st.final_output.label is not None and (
                not escalated or all(getattr(st, f'{s}_output') and getattr(st, f'{s}_output').model_output.label
                                     for s in ACTIVE)))
            if ok:
                pred = st.final_output.label
                break
            time.sleep(min(5 + attempt * 5, 45))
        if pred is None:  # unrecoverable → primary label fallback (tracked, so we can report contamination)
            pred = r['pred_label']; fallbacks += 1
            print(f'    FALLBACK to primary for {sid} (persistent failure)', flush=True)
        recs.append(dict(sample_id=sid, true=true, pred=pred, escalated=escalated))
        if escalated:
            esc += 1
            if esc % 40 == 0:
                print(f'      escalated {esc} (row {i+1})', flush=True)
            time.sleep(0.4)
    print(f'    order done: {fallbacks} fallback(s)', flush=True)
    return recs, fallbacks


def metrics(recs):
    yt = [r['true'] for r in recs]; yp = [r['pred'] for r in recs]
    esc = [r for r in recs if r['escalated']]
    return dict(acc=accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average='macro'),
                esc_acc=accuracy_score([r['true'] for r in esc], [r['pred'] for r in esc]),
                esc_correct=sum(1 for r in esc if r['pred'] == r['true']), n_esc=len(esc))


all_recs = {}
summary = []
for name, order in ORDERS:
    print(f'=== {name} ===', flush=True)
    recs, fallbacks = run_order(order)
    all_recs[name] = recs
    m = metrics(recs); m['name'] = name; m['fallbacks'] = fallbacks
    summary.append(m)
    # incremental save so a later crash cannot lose completed orders
    json.dump(all_recs, open(OUTDIR + '/records.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

o = io.StringIO()
def P(s=''): o.write(s + '\n')
P('SEQUENTIAL AGENT-ORDER SWEEP — Design C, Ahmed, %s, threshold %.2f' % (MODEL, THRESH))
P('sequential_chain=True; only agent_stage_order varies. 818 samples, %d escalated.' % summary[0]['n_esc'])
P('(For reference: parallel Design C ~0.9211 F1; the default seq order is the L->P->C row.)')
P('')
P('%-20s | %-8s | %-9s | %-16s | %s' % ('order', 'acc', 'macro F1', 'escalated', 'fallbacks'))
P('-' * 72)
for m in summary:
    P('%-20s | %.4f   | %.4f    | %d/%d (%.4f) | %d' %
      (m['name'], m['acc'], m['macro_f1'], m['esc_correct'], m['n_esc'], m['esc_acc'], m['fallbacks']))
f1s = [m['macro_f1'] for m in summary]
P('')
P('macro-F1 spread across orders: %.4f  (max %.4f - min %.4f)' % (max(f1s) - min(f1s), max(f1s), min(f1s)))
open(OUTDIR + '/order_sweep_report.txt', 'w', encoding='utf-8').write(o.getvalue())
print('\n' + o.getvalue()); print('saved ->', OUTDIR, flush=True)
