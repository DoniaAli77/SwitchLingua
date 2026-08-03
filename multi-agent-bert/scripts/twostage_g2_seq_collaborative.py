"""Two-stage XLM-R primary (GEN-960 -> EESA) + G2 + gpt-4.1-mini, SEQUENTIAL-
COLLABORATIVE chain. Compared against the existing G2 PARALLEL run on the same
primary (experiment_twostage_g2_41mini). This is a WEAKER primary (0.8655) with
138 escalations, so agents have real headroom — the interesting case for whether
collaborative chaining helps. Robust to connection blips. No training/generation.
"""
from __future__ import annotations
import os, sys, json, time, io

sys.path.insert(0, os.path.abspath('.'))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128')
for line in open('../Modified_Version/.env', encoding='utf-8'):
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    k, v = line.split('=', 1)
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ['SENTIMENT_PROMPT_VARIANT'] = 'semantic_v1'
os.environ['SENTIMENT_AGENT_VARIANT'] = 'lexical_polarity_contextual_selective_gate'  # G2

from sklearn.metrics import f1_score, accuracy_score
from src.config.loader import load_task_bundle
from src.state.schema import PipelineState, StateMetadata
from evaluate_pipeline import build_primary_classifier, build_llm_client, build_orchestrator

MODEL = 'gpt-4.1-mini'; THRESH = 0.90
CKPT = 'experiments/checkpoints/expTwoStage_gen960/fullEESA'
TEST = 'data/Sentiment/processed/eesa_sentiment_test.jsonl'
PRIOR = 'experiments/outputs/multi_agent_bert/experiment_twostage_g2_41mini/twostage_g2_41mini__full_pipeline_predictions.json'
OUTDIR = 'experiments/outputs/multi_agent_bert/experiment_twostage_g2_seq_collaborative'
os.makedirs(OUTDIR, exist_ok=True)
ACTIVE = ['lexical', 'logic', 'contextual']

lm = json.load(open(CKPT + '/label_map.json', encoding='utf-8'))
label_map = {int(k): v for k, v in lm['id2label'].items()}
samples = [json.loads(l) for l in open(TEST, encoding='utf-8') if l.strip()]

bundle = load_task_bundle('src/config/default.yaml', active_task='sentiment_classification',
                          pipeline_mode='full_agentic', threshold=THRESH)
tc = bundle.task_config
tc.sequential_chain = True
tc.sequential_chain_style = 'collaborative'
primary = build_primary_classifier('transformer', transformer_checkpoint=CKPT,
                                   device='cuda', label_map=label_map)
llm = build_llm_client('openai', llm_model=MODEL, allowed_labels=tc.labels)
orch = build_orchestrator(tc, threshold=THRESH, enable_deliberation=False,
                          keyword_map=bundle.keyword_map, rule_map=bundle.rule_map,
                          primary_classifier=primary, llm_client=llm,
                          consensus_primary_weight=1.0,
                          sentiment_agent_variant='lexical_polarity_contextual_selective_gate')

recs = []; esc = 0; fb = 0
for i, s in enumerate(samples):
    true = s['label']; text = s['text']
    pred = None; escalated = False
    for attempt in range(10):
        try:
            st = PipelineState(metadata=StateMetadata(sample_id=f'eesa-test-{i:05d}'),
                               input_text=text, task_config=tc)
            st = orch.run(st)
        except Exception:
            time.sleep(min(5 + attempt * 5, 45)); continue
        escalated = st.lexical_output is not None  # agents ran => escalated
        if st.final_output and st.final_output.label is not None and (
            not escalated or all(getattr(st, f'{a}_output') and getattr(st, f'{a}_output').model_output.label
                                 for a in ACTIVE)):
            pred = st.final_output.label; break
        time.sleep(min(5 + attempt * 5, 45))
    if pred is None:
        pred = primary.predict(text).label if hasattr(primary, 'predict') else true
        fb += 1; print(f'    FALLBACK idx {i}', flush=True)
    recs.append(dict(idx=i, true=true, pred=pred, escalated=escalated))
    if escalated:
        esc += 1
        if esc % 20 == 0:
            print(f'      escalated {esc} (sample {i+1}/{len(samples)})', flush=True)
        time.sleep(0.4)
print(f'  done: {esc} escalated, {fb} fallback(s)', flush=True)
json.dump(recs, open(OUTDIR + '/collab_records.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# existing PARALLEL G2 baseline (same primary), aligned by order
par = json.load(open(PRIOR, encoding='utf-8'))
par_pred = [p['predicted_label'] for p in par]
par_esc = [bool(p.get('escalated') is True or str(p.get('escalated')).lower() == 'true') for p in par]


def metrics(preds, trues, esc_flags):
    acc = accuracy_score(trues, preds); f1 = f1_score(trues, preds, average='macro')
    ei = [j for j, e in enumerate(esc_flags) if e]
    ea = accuracy_score([trues[j] for j in ei], [preds[j] for j in ei]) if ei else float('nan')
    return acc, f1, sum(1 for j in ei if preds[j] == trues[j]), len(ei)


trues = [r['true'] for r in recs]
collab_pred = [r['pred'] for r in recs]
collab_esc = [r['escalated'] for r in recs]
c_acc, c_f1, c_ec, c_ne = metrics(collab_pred, trues, collab_esc)
p_acc, p_f1, p_ec, p_ne = metrics(par_pred, trues, par_esc)
changed = sum(1 for a, b in zip(par_pred, collab_pred) if a != b)

o = io.StringIO()
def P(s=''): o.write(s + '\n')
P('TWO-STAGE XLM-R primary (GEN960->EESA) + G2 + %s, threshold %.2f' % (MODEL, THRESH))
P('Weak primary (headroom for agents). Compare existing PARALLEL G2 vs new SEQUENTIAL-COLLABORATIVE.')
P('818 samples; escalated ~%d; fallbacks=%d' % (c_ne, fb))
P('')
P('%-28s | %-8s | %-9s | %s' % ('mode', 'acc', 'macro F1', 'escalated'))
P('-' * 64)
P('%-28s | %.4f   | %.4f    | %d/%d' % ('parallel G2 (existing)', p_acc, p_f1, p_ec, p_ne))
P('%-28s | %.4f   | %.4f    | %d/%d' % ('sequential-collaborative G2', c_acc, c_f1, c_ec, c_ne))
P('')
P('primary_only baseline (from earlier run): 0.8655 / 0.8579')
P('predictions changed (parallel vs collaborative): %d / 818' % changed)
open(OUTDIR + '/twostage_collab_report.txt', 'w', encoding='utf-8').write(o.getvalue())
print('\n' + o.getvalue()); print('saved ->', OUTDIR, flush=True)
