"""Polarity-weight sweep (Design C, NO gate) — can upweighting Polarity substitute
for the Selective IntentGate?

Design H showed the gate's criteria are inert inside a vote at weight 1.0. The
natural follow-up: give the Polarity agent MORE voting power so it can outvote the
other two specialists on meta/mention cases, and see whether that recovers the
gate's benefit.

Method: run Design C once (Lexical + Polarity@logic + Contextual, no gate) capturing
each agent's (label, confidence) plus the primary's softmax, then re-fuse OFFLINE
with the production ConsensusAgent at a range of Polarity ("logic" slot) weights.
One paid pass yields the whole curve. Non-escalated samples are unaffected by
weights (they never reach consensus).

Baselines: primary_only 0.9254/0.9207 (63/84); Design C @w=1.0 0.9266/0.9216 (64/84);
Design G2 (gate) mean n=5 0.9306/0.9266 (67.2/84), noise band 0.9291-0.9315.
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
os.environ['SENTIMENT_AGENT_VARIANT'] = 'lexical_polarity_contextual'  # Design C, no gate

from sklearn.metrics import f1_score, accuracy_score
from src.config.loader import load_task_bundle
from src.state.schema import PipelineState, StateMetadata, AgentOutput, ModelOutput
from src.agents.consensus_agent import ConsensusAgent
from evaluate_pipeline import build_primary_classifier, build_llm_client, build_orchestrator

MODEL = 'gpt-4.1-mini'; THRESH = 0.70
VARIANT = 'lexical_polarity_contextual'
ALIGNED = 'data/Sentiment/external/ahmed/ahmed_eesa_test_predictions_aligned.csv'
OUTDIR = 'experiments/outputs/multi_agent_bert/experiment_ahmed_polarity_weight_sweep'
os.makedirs(OUTDIR, exist_ok=True)
VOTERS = ['lexical', 'logic', 'contextual']
WEIGHTS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
rows = list(csv.DictReader(open(ALIGNED, encoding='utf-8')))

bundle = load_task_bundle('src/config/default.yaml', active_task='sentiment_classification',
                          pipeline_mode='full_agentic', threshold=THRESH)
tc = bundle.task_config
LABELS = tc.labels
primary = build_primary_classifier('precomputed', precomputed_predictions=ALIGNED)
llm = build_llm_client('openai', llm_model=MODEL, allowed_labels=LABELS)
orch = build_orchestrator(tc, threshold=THRESH, enable_deliberation=False,
                          keyword_map=bundle.keyword_map, rule_map=bundle.rule_map,
                          primary_classifier=primary, llm_client=llm,
                          consensus_primary_weight=1.0, sentiment_agent_variant=VARIANT)

# ---- one paid pass: capture agent outputs on escalated samples ----------------
captured = []
esc = 0; fb = 0
for i, r in enumerate(rows):
    sid = r['sample_id']; true = r['true_label']; text = r['text']
    escalated = float(r['confidence']) < THRESH
    pmo = dict(label=r['pred_label'], confidence=float(r['confidence']),
               probabilities={'positive': float(r['prob_positive']),
                              'negative': float(r['prob_negative']),
                              'neutral': float(r['prob_neutral'])})
    if not escalated:
        captured.append(dict(sample_id=sid, true=true, escalated=False,
                             primary=pmo, agents=None))
        continue
    agents = None
    for attempt in range(10):
        try:
            st = PipelineState(metadata=StateMetadata(sample_id=sid), input_text=text, task_config=tc)
            st = orch.run(st)
        except Exception:
            time.sleep(min(5 + attempt * 5, 45)); continue
        if all(getattr(st, f'{a}_output') and getattr(st, f'{a}_output').model_output.label
               for a in VOTERS):
            agents = {a: dict(label=getattr(st, f'{a}_output').model_output.label,
                              confidence=float(getattr(st, f'{a}_output').model_output.confidence))
                      for a in VOTERS}
            break
        time.sleep(min(5 + attempt * 5, 45))
    if agents is None:
        fb += 1; print(f'    FALLBACK {sid}', flush=True)
    captured.append(dict(sample_id=sid, true=true, escalated=True, primary=pmo, agents=agents))
    esc += 1
    if esc % 20 == 0:
        print(f'      escalated {esc}', flush=True)
    time.sleep(0.4)
print(f'  capture done: {esc} escalated, {fb} fallback(s)', flush=True)
json.dump(captured, open(OUTDIR + '/captured.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

# ---- offline re-fusion at each Polarity ("logic" slot) weight -----------------
def refuse(rec, w_logic):
    """Final label under Design C consensus with logic-slot weight = w_logic."""
    if not rec['escalated'] or rec['agents'] is None:
        return rec['primary']['label']
    st = PipelineState(metadata=StateMetadata(sample_id='x'), input_text='', task_config=tc)
    for slot in VOTERS:
        a = rec['agents'][slot]
        setattr(st, f'{slot}_output',
                AgentOutput(agent_name=slot,
                            model_output=ModelOutput(label=a['label'], confidence=a['confidence'])))
    p = rec['primary']
    st.primary_model_output = ModelOutput(label=p['label'], confidence=p['confidence'],
                                          probabilities=p['probabilities'])
    cons = ConsensusAgent(weights={'primary': 1.0, 'logic': w_logic})
    cons.run(st)
    return st.final_output.label


trues = [c['true'] for c in captured]
esc_idx = [j for j, c in enumerate(captured) if c['escalated']]
results = []
for w in WEIGHTS:
    preds = [refuse(c, w) for c in captured]
    acc = accuracy_score(trues, preds); f1 = f1_score(trues, preds, average='macro')
    ec = sum(1 for j in esc_idx if preds[j] == trues[j])
    results.append((w, acc, f1, ec))

o = io.StringIO()
def P(s=''): o.write(s + '\n')
P('POLARITY-WEIGHT SWEEP — Design C (no gate), Ahmed, %s, threshold %.2f' % (MODEL, THRESH))
P('Same captured agent outputs re-fused at each weight. 818 samples, %d escalated, %d fallback(s).'
  % (len(esc_idx), fb))
P('')
P('%-18s | %-8s | %-9s | %s' % ('Polarity weight', 'acc', 'macro F1', 'escalated'))
P('-' * 56)
P('%-18s | %.4f   | %.4f    | %d/84' % ('primary_only', 0.9254, 0.9207, 63))
for w, acc, f1, ec in results:
    tag = ' (Design C)' if w == 1.0 else ''
    P('%-18s | %.4f   | %.4f    | %d/%d' % ('w = %.2f%s' % (w, tag), acc, f1, ec, len(esc_idx)))
P('-' * 56)
P('%-18s | %.4f   | %.4f    | %s' % ('G2 gate (mean n=5)', 0.9306, 0.9266, '67.2/84'))
P('%-18s | %s | %s | %s' % ('G2 noise band', '0.9291-0.9315', '0.9251-0.9277', '66-68/84'))
best = max(results, key=lambda t: t[2])
P('')
P('best weight by macro F1: w=%.2f -> %.4f acc / %.4f F1 / %d escalated' % (best[0], best[1], best[2], best[3]))
P('reaches G2 band (>=0.9251 F1 and >=66 escalated)? %s'
  % ('YES' if (best[2] >= 0.9251 and best[3] >= 66) else 'NO'))
P('')
P('READ: if no weight reaches the G2 band, upweighting a voter cannot substitute for')
P('      the asymmetric post-consensus veto - upweighting is symmetric (Polarity wins')
P('      more often when right AND when wrong), whereas the gate only blocks moves')
P('      away from the primary.')
open(OUTDIR + '/polarity_weight_sweep_report.txt', 'w', encoding='utf-8').write(o.getvalue())
print('\n' + o.getvalue()); print('saved ->', OUTDIR, flush=True)
