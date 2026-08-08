"""G2 repeat runs — establish the temperature-0 run-to-run noise band.

Executes the IDENTICAL Design-G2 configuration 3 times (Ahmed precomputed primary,
selective gate, gpt-4.1-mini, threshold 0.70, semantic_v1, w_primary 1.0) and reports
each run plus the mean/range. Combined with the two existing measurements (canonical
0.9303 and the fresh 0.9315) this gives n=5, so the thesis can quote a noise band and
justify treating +-1-sample differences as non-significant. No training/generation.
"""
from __future__ import annotations
import os, sys, json, csv, time, io, statistics

sys.path.insert(0, os.path.abspath('.'))
for line in open('../Modified_Version/.env', encoding='utf-8'):
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    k, v = line.split('=', 1)
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ['SENTIMENT_PROMPT_VARIANT'] = 'semantic_v1'
os.environ['SENTIMENT_AGENT_VARIANT'] = 'lexical_polarity_contextual_selective_gate'

from sklearn.metrics import f1_score, accuracy_score
from src.config.loader import load_task_bundle
from src.state.schema import PipelineState, StateMetadata
from evaluate_pipeline import build_primary_classifier, build_llm_client, build_orchestrator

MODEL = 'gpt-4.1-mini'; THRESH = 0.70
VARIANT = 'lexical_polarity_contextual_selective_gate'
ALIGNED = 'data/Sentiment/external/ahmed/ahmed_eesa_test_predictions_aligned.csv'
OUTDIR = 'experiments/outputs/multi_agent_bert/experiment_ahmed_g2_repeats'
os.makedirs(OUTDIR, exist_ok=True)
ACTIVE = ['lexical', 'logic', 'contextual']
N_REPEATS = 3
rows = list(csv.DictReader(open(ALIGNED, encoding='utf-8')))

# Prior measurements of the SAME config (from earlier registered runs).
PRIOR = [
    ('run 1 (canonical)', 0.9303, 0.9262, 67),
    ('run 2 (fresh)', 0.9315, 0.9277, 68),
]


def one_run(tag):
    bundle = load_task_bundle('src/config/default.yaml', active_task='sentiment_classification',
                              pipeline_mode='full_agentic', threshold=THRESH)
    tc = bundle.task_config
    primary = build_primary_classifier('precomputed', precomputed_predictions=ALIGNED)
    llm = build_llm_client('openai', llm_model=MODEL, allowed_labels=tc.labels)
    orch = build_orchestrator(tc, threshold=THRESH, enable_deliberation=False,
                              keyword_map=bundle.keyword_map, rule_map=bundle.rule_map,
                              primary_classifier=primary, llm_client=llm,
                              consensus_primary_weight=1.0, sentiment_agent_variant=VARIANT)
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
                not escalated or all(getattr(st, f'{a}_output') and getattr(st, f'{a}_output').model_output.label
                                     for a in ACTIVE)):
                pred = st.final_output.label; break
            time.sleep(min(5 + attempt * 5, 45))
        if pred is None:
            pred = r['pred_label']; fb += 1
        recs.append(dict(sample_id=sid, true=true, pred=pred, escalated=escalated))
        if escalated:
            esc += 1
            if esc % 40 == 0:
                print(f'      [{tag}] escalated {esc}', flush=True)
            time.sleep(0.4)
    yt = [x['true'] for x in recs]; yp = [x['pred'] for x in recs]
    e = [x for x in recs if x['escalated']]
    print(f'    [{tag}] done, {fb} fallback(s)', flush=True)
    return recs, (accuracy_score(yt, yp), f1_score(yt, yp, average='macro'),
                  sum(1 for x in e if x['pred'] == x['true']), len(e), fb)


all_recs = {}; new = []
for k in range(N_REPEATS):
    tag = f'repeat {k+1}'
    print(f'=== {tag}/{N_REPEATS} ===', flush=True)
    recs, m = one_run(tag)
    all_recs[tag] = recs
    new.append((f'run {len(PRIOR)+k+1} ({tag})', m[0], m[1], m[2]))
    json.dump(all_recs, open(OUTDIR + '/records.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

table = PRIOR + new
accs = [t[1] for t in table]; f1s = [t[2] for t in table]; escs = [t[3] for t in table]

o = io.StringIO()
def P(s=''): o.write(s + '\n')
P('DESIGN G2 — REPEATED RUNS (identical config) — Ahmed, %s, threshold %.2f' % (MODEL, THRESH))
P('Purpose: establish the temperature-0 run-to-run noise band. 818 samples, 84 escalated.')
P('')
P('%-22s | %-8s | %-9s | %s' % ('run', 'acc', 'macro F1', 'escalated'))
P('-' * 58)
for name, a, f, e in table:
    P('%-22s | %.4f   | %.4f    | %d/84' % (name, a, f, e))
P('-' * 58)
P('%-22s | %.4f   | %.4f    | %.1f/84' % ('MEAN (n=%d)' % len(table),
                                          statistics.mean(accs), statistics.mean(f1s), statistics.mean(escs)))
P('%-22s | %.4f   | %.4f    | %d-%d' % ('RANGE', max(accs) - min(accs), max(f1s) - min(f1s),
                                        min(escs), max(escs)))
if len(table) > 1:
    P('%-22s | %.4f   | %.4f    |' % ('STDEV', statistics.stdev(accs), statistics.stdev(f1s)))
P('')
P('min-max accuracy : %.4f - %.4f' % (min(accs), max(accs)))
P('min-max macro F1 : %.4f - %.4f' % (min(f1s), max(f1s)))
P('=> differences of +-%d escalated sample(s) are within run-to-run noise.' % (max(escs) - min(escs)))
open(OUTDIR + '/g2_repeats_report.txt', 'w', encoding='utf-8').write(o.getvalue())
print('\n' + o.getvalue()); print('saved ->', OUTDIR, flush=True)
