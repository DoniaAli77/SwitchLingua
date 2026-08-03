"""Does a COLLABORATIVE chain framing change the sequential result?

The earlier sequential runs used an anti-anchoring block ("do your own analysis
FIRST ... do NOT simply copy them"), which biases the chained agents toward
independence. This runs three modes on the SAME Design-C / Ahmed / gpt-4o-mini
setup so we can see whether inviting the agents to build on each other matters:

  1. parallel                 (sequential_chain=False)
  2. sequential-independent   (chain ON, anti-anchoring framing)
  3. sequential-collaborative (chain ON, "build on your teammates" framing)

Everything else fixed. Robust to transient connection errors (retry + primary
fallback, tracked). No training/generation.
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
OUTDIR = 'experiments/outputs/multi_agent_bert/experiment_ahmed_sequential_collaborative'
os.makedirs(OUTDIR, exist_ok=True)
ACTIVE = ['lexical', 'logic', 'contextual']
rows = list(csv.DictReader(open(ALIGNED, encoding='utf-8')))

MODES = [
    ("parallel",                 dict(sequential_chain=False, style="independent")),
    ("sequential-independent",   dict(sequential_chain=True,  style="independent")),
    ("sequential-collaborative", dict(sequential_chain=True,  style="collaborative")),
]


def run_mode(cfg):
    bundle = load_task_bundle('src/config/default.yaml', active_task='sentiment_classification',
                              pipeline_mode='full_agentic', threshold=THRESH)
    tc = bundle.task_config
    tc.sequential_chain = cfg['sequential_chain']
    tc.sequential_chain_style = cfg['style']
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
            pred = r['pred_label']; fb += 1
            print(f'    FALLBACK {sid}', flush=True)
        recs.append(dict(sample_id=sid, true=true, pred=pred, escalated=escalated))
        if escalated:
            esc += 1
            if esc % 40 == 0:
                print(f'      escalated {esc} (row {i+1})', flush=True)
            time.sleep(0.4)
    print(f'    done: {fb} fallback(s)', flush=True)
    return recs, fb


def metrics(recs):
    yt = [r['true'] for r in recs]; yp = [r['pred'] for r in recs]
    esc = [r for r in recs if r['escalated']]
    return dict(acc=accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average='macro'),
                esc_correct=sum(1 for r in esc if r['pred'] == r['true']), n_esc=len(esc))


all_recs = {}; summary = []
for name, cfg in MODES:
    print(f'=== {name} ===', flush=True)
    recs, fb = run_mode(cfg)
    all_recs[name] = recs
    m = metrics(recs); m['name'] = name; m['fb'] = fb
    summary.append(m)
    json.dump(all_recs, open(OUTDIR + '/records.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# changes vs parallel baseline
base = {r['sample_id']: r['pred'] for r in all_recs['parallel']}
def changes(recs): return sum(1 for r in recs if base[r['sample_id']] != r['pred'])

o = io.StringIO()
def P(s=''): o.write(s + '\n')
P('SEQUENTIAL FRAMING TEST — Design C, Ahmed, %s, threshold %.2f' % (MODEL, THRESH))
P('Same agents/prompts/primary/consensus; only chain on/off and framing vary. 818 samples, %d escalated.' % summary[0]['n_esc'])
P('')
P('%-26s | %-8s | %-9s | %-12s | %-10s | %s' % ('mode', 'acc', 'macro F1', 'escalated', 'vs parallel', 'fallbacks'))
P('-' * 88)
for m in summary:
    ch = changes(all_recs[m['name']])
    P('%-26s | %.4f   | %.4f    | %d/%d       | %-10s | %d' %
      (m['name'], m['acc'], m['macro_f1'], m['esc_correct'], m['n_esc'],
       ('baseline' if m['name'] == 'parallel' else f'{ch} changed'), m['fb']))
open(OUTDIR + '/collaborative_report.txt', 'w', encoding='utf-8').write(o.getvalue())
print('\n' + o.getvalue()); print('saved ->', OUTDIR, flush=True)
