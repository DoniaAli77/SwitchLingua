"""G2 — sequential-COLLABORATIVE only (parallel + seq-independent already run).

Runs just the collaborative-framing sequential mode on Design G (full gate,
gpt-4.1-mini), then combines with the existing parallel + sequential-independent
records from experiment_ahmed_parallel_vs_sequential_g2_41mini for the final
three-way table. Robust to transient connection errors. No training/generation.
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
os.environ['SENTIMENT_AGENT_VARIANT'] = 'lexical_polarity_contextual_selective_gate'  # Design G

from sklearn.metrics import f1_score, accuracy_score
from src.config.loader import load_task_bundle
from src.state.schema import PipelineState, StateMetadata
from evaluate_pipeline import build_primary_classifier, build_llm_client, build_orchestrator

MODEL = 'gpt-4.1-mini'; THRESH = 0.70
D = 'data/Sentiment/external/ahmed'
ALIGNED = D + '/ahmed_eesa_test_predictions_aligned.csv'
PRIOR = 'experiments/outputs/multi_agent_bert/experiment_ahmed_parallel_vs_sequential_g2_41mini/records.json'
OUTDIR = 'experiments/outputs/multi_agent_bert/experiment_ahmed_sequential_collaborative_g2'
os.makedirs(OUTDIR, exist_ok=True)
ACTIVE = ['lexical', 'logic', 'contextual']
rows = list(csv.DictReader(open(ALIGNED, encoding='utf-8')))


def run_collaborative():
    bundle = load_task_bundle('src/config/default.yaml', active_task='sentiment_classification',
                              pipeline_mode='full_agentic', threshold=THRESH)
    tc = bundle.task_config
    tc.sequential_chain = True
    tc.sequential_chain_style = 'collaborative'
    primary = build_primary_classifier('precomputed', precomputed_predictions=ALIGNED)
    llm = build_llm_client('openai', llm_model=MODEL, allowed_labels=tc.labels)
    orch = build_orchestrator(tc, threshold=THRESH, enable_deliberation=False,
                              keyword_map=bundle.keyword_map, rule_map=bundle.rule_map,
                              primary_classifier=primary, llm_client=llm,
                              consensus_primary_weight=1.0,
                              sentiment_agent_variant='lexical_polarity_contextual_selective_gate')
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
            if esc % 20 == 0:
                print(f'      escalated {esc} (row {i+1})', flush=True)
            time.sleep(0.4)
    print(f'    done: {fb} fallback(s)', flush=True)
    return recs, fb


def metrics(recs):
    yt = [r['true'] for r in recs]; yp = [r['pred'] for r in recs]
    esc = [r for r in recs if r['escalated']]
    return dict(acc=accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average='macro'),
                esc_correct=sum(1 for r in esc if r['pred'] == r['true']), n_esc=len(esc))


print('=== G2 — sequential-collaborative ===', flush=True)
collab, fb = run_collaborative()

prior = json.load(open(PRIOR, encoding='utf-8'))
all_recs = {
    'parallel': prior['parallel'],
    'sequential-independent': prior['sequential'],
    'sequential-collaborative': collab,
}
json.dump(all_recs, open(OUTDIR + '/records.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

base = {r['sample_id']: r['pred'] for r in all_recs['parallel']}
def changes(recs): return sum(1 for r in recs if base[r['sample_id']] != r['pred'])

o = io.StringIO()
def P(s=''): o.write(s + '\n')
P('SEQUENTIAL FRAMING TEST — G2 (selective gate), Ahmed, %s, threshold %.2f' % (MODEL, THRESH))
P('Same agents/prompts/primary/gate/consensus; only chain on/off and framing vary. 818 samples, 84 escalated.')
P('(parallel + sequential-independent reused from the earlier Design-G run; collaborative is new; fallbacks=%d)' % fb)
P('')
P('%-26s | %-8s | %-9s | %-10s | %s' % ('mode', 'acc', 'macro F1', 'escalated', 'vs parallel'))
P('-' * 74)
for name in ('parallel', 'sequential-independent', 'sequential-collaborative'):
    m = metrics(all_recs[name])
    tag = 'baseline' if name == 'parallel' else f'{changes(all_recs[name])} changed'
    P('%-26s | %.4f   | %.4f    | %d/%d      | %s' %
      (name, m['acc'], m['macro_f1'], m['esc_correct'], m['n_esc'], tag))
open(OUTDIR + '/collaborative_g2_report.txt', 'w', encoding='utf-8').write(o.getvalue())
print('\n' + o.getvalue()); print('saved ->', OUTDIR, flush=True)
