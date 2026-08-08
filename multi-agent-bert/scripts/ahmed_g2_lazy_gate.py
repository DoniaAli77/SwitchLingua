"""G2-lazy — the selective gate's LLM call DEFERRED to the post-consensus stage.

Identical to G2 except the IntentGate agent is not a pre-consensus specialist: it is
invoked by the post-consensus guard only when consensus actually overrode the primary.
Because the gate's prompt never sees the consensus or primary prediction, its judgement
is unchanged — so decisions should match G2 within run-to-run noise while the gate's
LLM calls drop to the override subset only.

Compare against the G2 noise band (n=5): acc 0.9291-0.9315 (mean 0.9306),
macro F1 0.9251-0.9277 (mean 0.9266), escalated 66-68/84.
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
os.environ['SENTIMENT_AGENT_VARIANT'] = 'lexical_polarity_contextual_lazy_gate'

from sklearn.metrics import f1_score, accuracy_score
from src.config.loader import load_task_bundle
from src.state.schema import PipelineState, StateMetadata
from evaluate_pipeline import build_primary_classifier, build_llm_client, build_orchestrator

MODEL = 'gpt-4.1-mini'; THRESH = 0.70
VARIANT = 'lexical_polarity_contextual_lazy_gate'
ALIGNED = 'data/Sentiment/external/ahmed/ahmed_eesa_test_predictions_aligned.csv'
OUTDIR = 'experiments/outputs/multi_agent_bert/experiment_ahmed_g2_lazy_gate'
os.makedirs(OUTDIR, exist_ok=True)
VOTERS = ['lexical', 'logic', 'contextual']
rows = list(csv.DictReader(open(ALIGNED, encoding='utf-8')))

bundle = load_task_bundle('src/config/default.yaml', active_task='sentiment_classification',
                          pipeline_mode='full_agentic', threshold=THRESH)
tc = bundle.task_config
primary = build_primary_classifier('precomputed', precomputed_predictions=ALIGNED)
llm = build_llm_client('openai', llm_model=MODEL, allowed_labels=tc.labels)
orch = build_orchestrator(tc, threshold=THRESH, enable_deliberation=False,
                          keyword_map=bundle.keyword_map, rule_map=bundle.rule_map,
                          primary_classifier=primary, llm_client=llm,
                          consensus_primary_weight=1.0, sentiment_agent_variant=VARIANT)

recs = []; esc = 0; fb = 0; gate_calls = 0; gate_fired = 0
for i, r in enumerate(rows):
    sid = r['sample_id']; true = r['true_label']; text = r['text']
    escalated = float(r['confidence']) < THRESH
    pred = None; called = False; fired = False
    for attempt in range(10):
        try:
            st = PipelineState(metadata=StateMetadata(sample_id=sid), input_text=text, task_config=tc)
            st = orch.run(st)
        except Exception:
            time.sleep(min(5 + attempt * 5, 45)); continue
        if st.final_output and st.final_output.label is not None and (
            not escalated or all(getattr(st, f'{a}_output') and getattr(st, f'{a}_output').model_output.label
                                 for a in VOTERS)):
            pred = st.final_output.label
            # gate LLM was invoked iff the polarity slot got populated (lazy mode)
            called = st.polarity_output is not None
            fired = 'intent_gate=BLOCKED' in ((st.consensus_output.rationale or '')
                                              if st.consensus_output else '')
            break
        time.sleep(min(5 + attempt * 5, 45))
    if pred is None:
        pred = r['pred_label']; fb += 1
        print(f'    FALLBACK {sid}', flush=True)
    recs.append(dict(sample_id=sid, true=true, pred=pred, escalated=escalated,
                     gate_called=called, gate_fired=fired))
    if escalated:
        esc += 1
        gate_calls += int(called); gate_fired += int(fired)
        if esc % 20 == 0:
            print(f'      escalated {esc}, gate calls so far {gate_calls}', flush=True)
        time.sleep(0.4)
print(f'  done: {esc} escalated, {gate_calls} gate calls, {gate_fired} fired, {fb} fallback(s)', flush=True)
json.dump(recs, open(OUTDIR + '/records.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

yt = [x['true'] for x in recs]; yp = [x['pred'] for x in recs]
e = [x for x in recs if x['escalated']]
acc = accuracy_score(yt, yp); f1 = f1_score(yt, yp, average='macro')
ec = sum(1 for x in e if x['pred'] == x['true'])

o = io.StringIO()
def P(s=''): o.write(s + '\n')
P('G2-LAZY — selective gate invoked ONLY after consensus, ONLY on overrides')
P('Ahmed precomputed primary, %s, threshold %.2f, semantic_v1, w_primary 1.0' % (MODEL, THRESH))
P('818 samples, %d escalated, %d fallback(s)' % (len(e), fb))
P('')
P('%-26s | %-8s | %-9s | %-11s | %s' % ('config', 'acc', 'macro F1', 'escalated', 'gate LLM calls'))
P('-' * 80)
P('%-26s | %.4f   | %.4f    | %-11s | %s' % ('G2 (eager) mean n=5', 0.9306, 0.9266, '67.2/84', '84 (always)'))
P('%-26s | %.4f-%.4f | %.4f-%.4f | %-11s | %s' % ('G2 range (noise band)', 0.9291, 0.9315, 0.9251, 0.9277, '66-68/84', '84'))
P('%-26s | %.4f   | %.4f    | %-11s | %d (%.0f%% saved)' % (
    'G2-LAZY (this run)', acc, f1, '%d/%d' % (ec, len(e)), gate_calls,
    100.0 * (len(e) - gate_calls) / len(e) if e else 0.0))
P('')
P('gate invoked on %d/%d escalated samples; blocked an override on %d' % (gate_calls, len(e), gate_fired))
inside = (0.9291 <= acc <= 0.9315)
P('accuracy inside G2 noise band (0.9291-0.9315)? %s' % ('YES' if inside else 'NO'))
P('')
P('READ: if inside the band, deferring the gate call is decision-equivalent to G2 and')
P('      strictly cheaper - and the "gate runs after consensus" description becomes literal.')
open(OUTDIR + '/g2_lazy_report.txt', 'w', encoding='utf-8').write(o.getvalue())
print('\n' + o.getvalue()); print('saved ->', OUTDIR, flush=True)
