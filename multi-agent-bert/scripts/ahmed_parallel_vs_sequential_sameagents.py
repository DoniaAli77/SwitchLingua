"""Fair parallel-vs-sequential comparison with the SAME agents (Ahmed, Design C).

Holds everything constant — the exact three specialists (Lexical + Polarity +
Contextual), the same prompts, the same Ahmed precomputed primary, threshold 0.70,
gpt-4o-mini, and the same consensus vote — and flips ONE switch:
``task_config.sequential_chain``. Parallel mode = agents independent; Sequential
mode = agents run in order, each later agent sees the earlier specialists'
conclusions (agent-only chain block, no primary signal). Isolates the effect of
inter-agent ordering alone. No training, no generation.
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
os.environ['SENTIMENT_AGENT_VARIANT'] = 'lexical_polarity_contextual'  # Design C trio

from sklearn.metrics import f1_score, accuracy_score
from src.config.loader import load_task_bundle
from src.state.schema import PipelineState, StateMetadata
from evaluate_pipeline import build_primary_classifier, build_llm_client, build_orchestrator

MODEL = 'gpt-4o-mini'
THRESH = 0.70
D = 'data/Sentiment/external/ahmed'
ALIGNED = D + '/ahmed_eesa_test_predictions_aligned.csv'
OUTDIR = 'experiments/outputs/multi_agent_bert/experiment_ahmed_parallel_vs_sequential_sameagents'
os.makedirs(OUTDIR, exist_ok=True)
ACTIVE = ['lexical', 'logic', 'contextual']
rows = list(csv.DictReader(open(ALIGNED, encoding='utf-8')))


def run_mode(sequential_chain: bool):
    bundle = load_task_bundle('src/config/default.yaml', active_task='sentiment_classification',
                              pipeline_mode='full_agentic', threshold=THRESH)
    tc = bundle.task_config
    tc.sequential_chain = sequential_chain  # the ONLY difference
    primary = build_primary_classifier('precomputed', precomputed_predictions=ALIGNED)
    llm = build_llm_client('openai', llm_model=MODEL, allowed_labels=tc.labels)
    orch = build_orchestrator(tc, threshold=THRESH, enable_deliberation=False,
                              keyword_map=bundle.keyword_map, rule_map=bundle.rule_map,
                              primary_classifier=primary, llm_client=llm,
                              consensus_primary_weight=1.0,
                              sentiment_agent_variant='lexical_polarity_contextual')
    recs = []
    esc = 0
    for i, r in enumerate(rows):
        sid = r['sample_id']; true = r['true_label']; text = r['text']
        escalated = float(r['confidence']) < THRESH
        st = None
        for _ in range(6):
            st = PipelineState(metadata=StateMetadata(sample_id=sid), input_text=text, task_config=tc)
            st = orch.run(st)
            if st.final_output and st.final_output.label is not None and (
                not escalated or all(getattr(st, f'{s}_output') and getattr(st, f'{s}_output').model_output.label
                                     for s in ACTIVE)):
                break
            time.sleep(10)
        recs.append(dict(sample_id=sid, true=true, pred=st.final_output.label, escalated=escalated))
        if escalated:
            esc += 1
            if esc % 20 == 0:
                print(f'    [{"seq" if sequential_chain else "par"}] escalated {esc} (row {i+1})', flush=True)
        if escalated:
            time.sleep(0.5)
    return recs


def metrics(recs):
    y_true = [r['true'] for r in recs]; y_pred = [r['pred'] for r in recs]
    esc = [r for r in recs if r['escalated']]
    return dict(
        acc=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average='macro'),
        esc_acc=accuracy_score([r['true'] for r in esc], [r['pred'] for r in esc]) if esc else float('nan'),
        n=len(recs), n_esc=len(esc),
    )


print('=== PARALLEL (sequential_chain=False) ===', flush=True)
par = run_mode(False)
print('=== SEQUENTIAL (sequential_chain=True) ===', flush=True)
seq = run_mode(True)

json.dump({'parallel': par, 'sequential': seq},
          open(OUTDIR + '/records.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
mp, ms = metrics(par), metrics(seq)
flips = sum(1 for a, b in zip(par, seq) if a['pred'] != b['pred'])

o = io.StringIO()
def P(s=''): o.write(s + '\n')
P('PARALLEL vs SEQUENTIAL — SAME AGENTS (Ahmed, Design C, %s, threshold %.2f)' % (MODEL, THRESH))
P('Only difference: task_config.sequential_chain (False vs True). 818 samples, %d escalated.' % mp['n_esc'])
P('prediction changes (parallel vs sequential): %d / 818' % flips)
P('')
P('%-22s | %-10s | %-10s' % ('metric', 'PARALLEL', 'SEQUENTIAL'))
P('-' * 50)
P('%-22s | %.4f     | %.4f' % ('accuracy (818)', mp['acc'], ms['acc']))
P('%-22s | %.4f     | %.4f' % ('macro F1 (818)', mp['macro_f1'], ms['macro_f1']))
P('%-22s | %.4f     | %.4f' % ('escalated accuracy', mp['esc_acc'], ms['esc_acc']))
open(OUTDIR + '/parallel_vs_sequential_report.txt', 'w', encoding='utf-8').write(o.getvalue())
print('\n' + o.getvalue())
print('saved ->', OUTDIR, flush=True)
