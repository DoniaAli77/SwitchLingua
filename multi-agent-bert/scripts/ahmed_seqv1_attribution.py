"""Decision-trace capture for sequential_sentiment_v1 on the Ahmed frozen-primary
full_agentic@0.7 escalated subset. Re-runs the 84 escalated samples through the
staged pipeline and records the per-sample sequential trace from state.extras:
decided_by rule, fallback_path, per-stage coercion flags, and stage_events
(retry / coerced_default / llm_error). No training, no generation. Temp 0.
"""
from __future__ import annotations
import os, sys, json, glob, csv, collections, time

sys.path.insert(0, os.path.abspath('.'))

for line in open('../Modified_Version/.env', encoding='utf-8'):
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    k, v = line.split('=', 1)
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ['SENTIMENT_AGENT_VARIANT'] = 'sequential_sentiment_v1'

from src.config.loader import load_task_bundle
from src.state.schema import PipelineState, StateMetadata
from src.agents.sequential_sentiment import SEQ_KEY
from evaluate_pipeline import build_primary_classifier, build_llm_client, build_orchestrator

D = 'data/Sentiment/external/ahmed'
RUN = 'experiments/outputs/multi_agent_bert/experiment_seqv1_ahmed'
OUT = RUN + '/decision_trace'
os.makedirs(OUT, exist_ok=True)

pr = json.load(open(glob.glob(RUN + '/*full_pipeline_predictions.json')[0], encoding='utf-8'))
esc = [x for x in pr if str(x.get('escalated')).lower() == 'true' or x.get('escalated') is True]
print('sequential_sentiment_v1 — escalated:', len(esc), flush=True)

bundle = load_task_bundle('src/config/default.yaml', active_task='sentiment_classification',
                          pipeline_mode='full_agentic', threshold=0.7)
tc = bundle.task_config
primary = build_primary_classifier('precomputed', precomputed_predictions=D + '/ahmed_eesa_test_predictions_aligned.csv')
llm = build_llm_client('openai', llm_model='gpt-4o-mini', allowed_labels=tc.labels)
orch = build_orchestrator(tc, threshold=0.7, enable_deliberation=False,
                          keyword_map=bundle.keyword_map, rule_map=bundle.rule_map,
                          primary_classifier=primary, llm_client=llm,
                          sentiment_agent_variant='sequential_sentiment_v1')

rows = []
for i, s in enumerate(esc):
    sid = s['sample_id']; text = s['input_text']; true = s['true_label']
    st = store = None
    for _ in range(5):
        st = PipelineState(metadata=StateMetadata(sample_id=sid), input_text=text, task_config=tc)
        st = orch.run(st)
        store = st.extras.get(SEQ_KEY, {})
        events = [e['event'] for e in store.get('stage_events', [])]
        if st.final_output and st.final_output.label is not None and 'llm_error' not in events:
            break
        time.sleep(12)
    intent = store.get('intent', {}); pol = store.get('polarity', {}); prag = store.get('pragmatic', {})
    events = store.get('stage_events', [])
    pm = st.primary_model
    f_lbl = st.final_output.label
    a_lbl = pm.label; a_ok = a_lbl == true; f_ok = f_lbl == true
    trans = ('correct_to_correct' if a_ok and f_ok else 'correct_to_wrong' if a_ok and not f_ok
             else 'wrong_to_correct' if not a_ok and f_ok else 'wrong_to_wrong')
    rows.append(dict(
        sample_id=sid, true_label=true, ahmed_label=a_lbl, ahmed_conf=round(pm.confidence, 4),
        decided_by=store.get('decided_by'), fallback_path=store.get('fallback_path'),
        intent_opinion=intent.get('opinion_expressed'), intent_conf=intent.get('confidence'),
        intent_use_mention=intent.get('use_vs_mention'),
        polarity_label=pol.get('label'), polarity_conf=pol.get('confidence'),
        pragmatic_decision=prag.get('keep_or_revise'), pragmatic_label=prag.get('final_label'),
        pragmatic_conf=prag.get('confidence'),
        intent_coerced=bool(intent.get('coerced')), polarity_coerced=bool(pol.get('coerced')),
        pragmatic_coerced=bool(prag.get('coerced')),
        n_retry=sum(1 for e in events if e['event'] == 'retry'),
        n_coerced=sum(1 for e in events if e['event'] == 'coerced_default'),
        n_llm_error=sum(1 for e in events if e['event'] == 'llm_error'),
        final_label=f_lbl, ahmed_correct=a_ok, final_correct=f_ok, transition=trans))
    if (i + 1) % 10 == 0: print('  done', i + 1, '/', len(esc), flush=True)
    time.sleep(0.5)

json.dump(rows, open(OUT + '/trace_table.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
with open(OUT + '/trace_table.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

n = len(rows)
print('\n--- decided_by distribution (of %d escalated) ---' % n)
for k, c in collections.Counter(r['decided_by'] for r in rows).most_common():
    print('   %-20s %d' % (k, c))
print('\n--- fallback_path (non-null) ---')
for k, c in collections.Counter(r['fallback_path'] for r in rows if r['fallback_path']).most_common():
    print('   %-24s %d' % (k, c))
print('\n--- per-stage coercion (safe-default fired) ---')
print('   intent   :', sum(1 for r in rows if r['intent_coerced']))
print('   polarity :', sum(1 for r in rows if r['polarity_coerced']))
print('   pragmatic:', sum(1 for r in rows if r['pragmatic_coerced']))
print('   samples with >=1 coerced stage:',
      sum(1 for r in rows if r['intent_coerced'] or r['polarity_coerced'] or r['pragmatic_coerced']))
print('\n--- retry / error events ---')
print('   total retries   :', sum(r['n_retry'] for r in rows))
print('   total coerced   :', sum(r['n_coerced'] for r in rows))
print('   total llm_errors:', sum(r['n_llm_error'] for r in rows))
print('\n--- pragmatic behaviour ---')
print('   revise:', sum(1 for r in rows if r['pragmatic_decision'] == 'revise'),
      ' keep:', sum(1 for r in rows if r['pragmatic_decision'] == 'keep'))
print('\n--- transitions ---')
print('  ', dict(collections.Counter(r['transition'] for r in rows)))
wc = sum(1 for r in rows if r['transition'] == 'wrong_to_correct')
cw = sum(1 for r in rows if r['transition'] == 'correct_to_wrong')
print('   W->C=%d  C->W=%d  net=%d' % (wc, cw, wc - cw))
print('   final escalated acc: %d/%d = %.4f' % (sum(1 for r in rows if r['final_correct']), n,
                                                sum(1 for r in rows if r['final_correct']) / n))
print('saved ->', OUT, flush=True)
