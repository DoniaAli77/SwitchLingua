"""Resume the COLLABORATIVE agent-order sweep: order 1 (L->P->C) already saved;
run the remaining 3 orders, merge, and write the full 4-order report."""
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
OUTDIR = 'experiments/outputs/multi_agent_bert/experiment_ahmed_sequential_order_sweep_collab'
ACTIVE = ['lexical', 'logic', 'contextual']
rows = list(csv.DictReader(open(ALIGNED, encoding='utf-8')))

REMAINING = [
    ("C->P->L (reverse)", ["contextual_agent", "logic_agent", "lexical_agent"]),
    ("P->L->C",           ["logic_agent", "lexical_agent", "contextual_agent"]),
    ("C->L->P",           ["contextual_agent", "lexical_agent", "logic_agent"]),
]
ORDER_KEYS = ["L->P->C (default)", "C->P->L (reverse)", "P->L->C", "C->L->P"]


def run_order(order):
    bundle = load_task_bundle('src/config/default.yaml', active_task='sentiment_classification',
                              pipeline_mode='full_agentic', threshold=THRESH)
    tc = bundle.task_config
    tc.sequential_chain = True
    tc.sequential_chain_style = 'collaborative'
    tc.agent_stage_order = order
    primary = build_primary_classifier('precomputed', precomputed_predictions=ALIGNED)
    llm = build_llm_client('openai', llm_model=MODEL, allowed_labels=tc.labels)
    orch = build_orchestrator(tc, threshold=THRESH, enable_deliberation=False,
                              keyword_map=bundle.keyword_map, rule_map=bundle.rule_map,
                              primary_classifier=primary, llm_client=llm,
                              consensus_primary_weight=1.0,
                              sentiment_agent_variant='lexical_polarity_contextual')
    recs = []; esc = 0; fb = 0
    for i, r in enumerate(rows):
        sid = r['sample_id']; true = r['true_label']; text = r['text']
        escalated = float(r['confidence']) < THRESH
        pred = None
        for attempt in range(10):
            try:
                st = PipelineState(metadata=StateMetadata(sample_id=sid), input_text=text, task_config=tc)
                st = orch.run(st)
            except Exception:
                time.sleep(min(5 + attempt * 5, 45)); continue
            if st.final_output and st.final_output.label is not None and (
                not escalated or all(getattr(st, f'{s}_output') and getattr(st, f'{s}_output').model_output.label
                                     for s in ACTIVE)):
                pred = st.final_output.label; break
            time.sleep(min(5 + attempt * 5, 45))
        if pred is None:
            pred = r['pred_label']; fb += 1; print(f'    FALLBACK {sid}', flush=True)
        recs.append(dict(sample_id=sid, true=true, pred=pred, escalated=escalated))
        if escalated:
            esc += 1
            if esc % 40 == 0:
                print(f'      escalated {esc} (row {i+1})', flush=True)
            time.sleep(0.4)
    print(f'    done: {fb} fallback(s)', flush=True)
    return recs, fb


all_recs = json.load(open(OUTDIR + '/records.json', encoding='utf-8'))  # has L->P->C
fbs = {ORDER_KEYS[0]: 0}
for name, order in REMAINING:
    print(f'=== {name} ===', flush=True)
    recs, fb = run_order(order)
    all_recs[name] = recs; fbs[name] = fb
    json.dump(all_recs, open(OUTDIR + '/records.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def metrics(recs):
    yt = [r['true'] for r in recs]; yp = [r['pred'] for r in recs]
    esc = [r for r in recs if r['escalated']]
    return (accuracy_score(yt, yp), f1_score(yt, yp, average='macro'),
            sum(1 for r in esc if r['pred'] == r['true']), len(esc))


o = io.StringIO()
def P(s=''): o.write(s + '\n')
P('SEQUENTIAL (COLLABORATIVE) AGENT-ORDER SWEEP — Design C, Ahmed, %s, threshold %.2f' % (MODEL, THRESH))
P('sequential_chain=True, style=collaborative; only agent_stage_order varies. 818 samples, 84 escalated.')
P('(For reference: parallel Design C ~0.9211 F1; independent-framing order sweep spread was 0.0021 F1.)')
P('')
P('%-20s | %-8s | %-9s | %-16s | %s' % ('order', 'acc', 'macro F1', 'escalated', 'fallbacks'))
P('-' * 72)
f1s = []
for name in ORDER_KEYS:
    acc, f1, ec, ne = metrics(all_recs[name]); f1s.append(f1)
    P('%-20s | %.4f   | %.4f    | %d/%d (%.4f) | %d' % (name, acc, f1, ec, ne, ec / ne, fbs.get(name, 0)))
P('')
P('macro-F1 spread across orders: %.4f  (max %.4f - min %.4f)' % (max(f1s) - min(f1s), max(f1s), min(f1s)))
open(OUTDIR + '/order_sweep_collab_report.txt', 'w', encoding='utf-8').write(o.getvalue())
print('\n' + o.getvalue()); print('saved ->', OUTDIR, flush=True)
