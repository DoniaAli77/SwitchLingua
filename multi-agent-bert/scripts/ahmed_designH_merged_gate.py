"""Design H — Selective-IntentGate criteria MERGED into the Polarity agent's prompt.

Ablation: is the gate's benefit from its PROMPT CONTENT or from its POSITION (a
non-voting post-consensus veto)? H = Lexical + Polarity(gate-merged prompt) +
Contextual, 3 voters, NO separate gate agent. Same Ahmed precomputed primary,
threshold 0.70, consensus w_primary 1.0, gpt-4.1-mini, semantic_v1 — so the only
change vs G2 is where the gate criteria live. No training/generation.

Baselines for comparison (same primary/model):
  primary_only            0.9254 / 0.9207   (63/84 escalated)
  Design C (no gate)      0.9266 / 0.9216   (64/84)
  Design G2 (gate)        0.9303 / 0.9262   (67/84)   [canonical]
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
os.environ['SENTIMENT_AGENT_VARIANT'] = 'lexical_polaritygate_contextual'  # Design H

from sklearn.metrics import f1_score, accuracy_score
from src.config.loader import load_task_bundle
from src.state.schema import PipelineState, StateMetadata
from evaluate_pipeline import build_primary_classifier, build_llm_client, build_orchestrator

MODEL = 'gpt-4.1-mini'; THRESH = 0.70
VARIANT = 'lexical_polaritygate_contextual'
ALIGNED = 'data/Sentiment/external/ahmed/ahmed_eesa_test_predictions_aligned.csv'
OUTDIR = 'experiments/outputs/multi_agent_bert/experiment_ahmed_designH_merged_gate'
os.makedirs(OUTDIR, exist_ok=True)
ACTIVE = ['lexical', 'logic', 'contextual']
rows = list(csv.DictReader(open(ALIGNED, encoding='utf-8')))

bundle = load_task_bundle('src/config/default.yaml', active_task='sentiment_classification',
                          pipeline_mode='full_agentic', threshold=THRESH)
tc = bundle.task_config
primary = build_primary_classifier('precomputed', precomputed_predictions=ALIGNED)
llm = build_llm_client('openai', llm_model=MODEL, allowed_labels=tc.labels)
orch = build_orchestrator(tc, threshold=THRESH, enable_deliberation=False,
                          keyword_map=bundle.keyword_map, rule_map=bundle.rule_map,
                          primary_classifier=primary, llm_client=llm,
                          consensus_primary_weight=1.0,
                          sentiment_agent_variant=VARIANT)

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
        print(f'    FALLBACK {sid}', flush=True)
    recs.append(dict(sample_id=sid, true=true, pred=pred, escalated=escalated,
                     primary=r['pred_label']))
    if escalated:
        esc += 1
        if esc % 20 == 0:
            print(f'      escalated {esc} (row {i+1})', flush=True)
        time.sleep(0.4)
print(f'  done: {esc} escalated, {fb} fallback(s)', flush=True)
json.dump(recs, open(OUTDIR + '/records.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

yt = [r['true'] for r in recs]; yp = [r['pred'] for r in recs]
e = [r for r in recs if r['escalated']]
acc = accuracy_score(yt, yp); f1 = f1_score(yt, yp, average='macro')
ec = sum(1 for r in e if r['pred'] == r['true'])
# attribution vs primary on escalated
w2c = sum(1 for r in e if r['primary'] != r['true'] and r['pred'] == r['true'])
c2w = sum(1 for r in e if r['primary'] == r['true'] and r['pred'] != r['true'])

o = io.StringIO()
def P(s=''): o.write(s + '\n')
P('DESIGN H — gate criteria MERGED into Polarity prompt (no separate gate)')
P('Ahmed precomputed primary, %s, threshold %.2f, semantic_v1, w_primary 1.0' % (MODEL, THRESH))
P('818 samples, %d escalated, %d fallback(s)' % (len(e), fb))
P('')
P('%-28s | %-8s | %-9s | %s' % ('config', 'acc', 'macro F1', 'escalated'))
P('-' * 66)
P('%-28s | %.4f   | %.4f    | %d/84' % ('primary_only', 0.9254, 0.9207, 63))
P('%-28s | %.4f   | %.4f    | %d/84' % ('Design C (no gate)', 0.9266, 0.9216, 64))
P('%-28s | %.4f   | %.4f    | %d/84' % ('Design G2 (gate, veto)', 0.9303, 0.9262, 67))
P('%-28s | %.4f   | %.4f    | %d/%d' % ('Design H (gate merged)', acc, f1, ec, len(e)))
P('')
P('Design H vs primary on escalated: wrong->correct=%d, correct->wrong=%d, net=%+d' % (w2c, c2w, w2c - c2w))
P('')
P('READ: if H ~= C (0.9216) the gate benefit comes from its POSITION (post-consensus veto).')
P('      if H ~= G2 (0.9262) the gate benefit comes from its PROMPT CONTENT.')
open(OUTDIR + '/designH_report.txt', 'w', encoding='utf-8').write(o.getvalue())
print('\n' + o.getvalue()); print('saved ->', OUTDIR, flush=True)
