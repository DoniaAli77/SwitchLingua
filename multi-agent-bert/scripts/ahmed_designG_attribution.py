"""Per-agent + gate-intervention capture for Design G (Lexical + Polarity +
Contextual + IntentGate) on the Ahmed frozen-primary full_agentic@0.7 escalated
subset (semantic_v1). Deterministic (temp 0) re-run capturing the four slots plus
the consensus rationale so we can count IntentGate interventions and classify each
as helped / hurt. No training, no generation.
"""
from __future__ import annotations
import os, sys, json, glob, csv, collections, re, time

for line in open('../Modified_Version/.env', encoding='utf-8'):
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    k, v = line.split('=', 1)
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ['SENTIMENT_PROMPT_VARIANT'] = 'semantic_v1'
os.environ['SENTIMENT_AGENT_VARIANT'] = 'lexical_polarity_contextual_intent_gate'

from src.config.loader import load_task_bundle
from src.state.schema import PipelineState, StateMetadata
from evaluate_pipeline import build_primary_classifier, build_llm_client, build_orchestrator

D = 'data/Sentiment/external/ahmed'
RUN = 'experiments/outputs/multi_agent_bert/experiment_ahmed_designG_intent_gate'
OUT = RUN + '/error_attribution'
os.makedirs(OUT, exist_ok=True)
ACTIVE = ['lexical', 'logic', 'contextual', 'polarity']  # polarity slot = IntentGate

pr = json.load(open(glob.glob(RUN + '/full/*predictions.json')[0], encoding='utf-8'))
esc = [x for x in pr if str(x.get('escalated')).lower() == 'true' or x.get('escalated') is True]
print('Design G — escalated:', len(esc), flush=True)

bundle = load_task_bundle('src/config/default.yaml', active_task='sentiment_classification',
                          pipeline_mode='full_agentic', threshold=0.7)
tc = bundle.task_config
primary = build_primary_classifier('precomputed', precomputed_predictions=D + '/ahmed_eesa_test_predictions_aligned.csv')
llm = build_llm_client('openai', llm_model='gpt-4o-mini', allowed_labels=tc.labels)
orch = build_orchestrator(tc, threshold=0.7, enable_deliberation=False,
                          keyword_map=bundle.keyword_map, rule_map=bundle.rule_map,
                          primary_classifier=primary, llm_client=llm,
                          consensus_primary_weight=1.0,
                          sentiment_agent_variant='lexical_polarity_contextual_intent_gate')

def lab(st, slot):
    o = getattr(st, f'{slot}_output')
    return o.model_output.label if o else None

_BLOCK = re.compile(r"intent_gate=BLOCKED override '([^']+)'->'([^']+)'")
rows = []
for i, s in enumerate(esc):
    sid = s['sample_id']; text = s['input_text']; true = s['true_label']
    st = None
    for _ in range(8):
        st = PipelineState(metadata=StateMetadata(sample_id=sid), input_text=text, task_config=tc)
        st = orch.run(st)
        if st.final_output and st.final_output.label is not None and all(lab(st, sl) is not None for sl in ACTIVE):
            break
        time.sleep(15)
    pm = st.primary_model
    rationale = st.consensus_output.rationale if st.consensus_output else ''
    m = _BLOCK.search(rationale or '')
    gate_blocked = bool(m)
    pre_gate = m.group(1) if m else None
    f_lbl = st.final_output.label
    a_lbl = pm.label; a_ok = a_lbl == true; f_ok = f_lbl == true
    trans = ('correct_to_correct' if a_ok and f_ok else 'correct_to_wrong' if a_ok and not f_ok
             else 'wrong_to_correct' if not a_ok and f_ok else 'wrong_to_wrong')
    rows.append(dict(sample_id=sid, text=text.replace('\n', ' ')[:120], true_label=true,
                     ahmed_label=a_lbl, ahmed_conf=round(pm.confidence, 4),
                     lexical_label=lab(st, 'lexical'), polarity_label=lab(st, 'logic'),
                     contextual_label=lab(st, 'contextual'), gate_label=lab(st, 'polarity'),
                     gate_blocked=gate_blocked, pre_gate_label=pre_gate,
                     final_label=f_lbl, ahmed_correct=a_ok, final_correct=f_ok, transition=trans))
    if (i + 1) % 10 == 0: print('  done', i + 1, '/', len(esc), flush=True)
    time.sleep(0.7)

json.dump(rows, open(OUT + '/attribution_table.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
with open(OUT + '/attribution_table.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

n = len(rows)
def acc(col):
    v = [r for r in rows if r[col] is not None]
    return round(sum(1 for r in v if r[col] == r['true_label']) / len(v), 4) if v else float('nan')
print('\n--- per-agent accuracy (escalated) ---')
for disp, col in [('lexical','lexical_label'),('polarity','polarity_label'),
                  ('contextual','contextual_label'),('intentgate','gate_label')]:
    print('   %-11s acc=%.4f' % (disp, acc(col)))
print('   %-11s acc=%.4f   ahmed=0.7500' % ('final', sum(1 for r in rows if r['final_correct'])/n))

# gate interventions
blk = [r for r in rows if r['gate_blocked']]
helped = [r for r in blk if r['pre_gate_label'] != r['true_label'] and r['final_label'] == r['true_label']]
hurt   = [r for r in blk if r['pre_gate_label'] == r['true_label'] and r['final_label'] != r['true_label']]
neutralchange = [r for r in blk if r not in helped and r not in hurt]
print('\n--- IntentGate interventions ---')
print('   total blocked overrides:', len(blk))
print('   helped (blocked a wrong polar over-call):', len(helped))
print('   hurt   (blocked a correct polar rescue):', len(hurt))
print('   no correctness change:', len(neutralchange))
# gate label behavior: how often gate==neutral, and accuracy of gate as a stance detector
gate_neu = sum(1 for r in rows if r['gate_label'] == 'neutral')
print('\n   gate said neutral (no-opinion) on %d/%d' % (gate_neu, n))
print('   transitions:', dict(collections.Counter(r['transition'] for r in rows)))
print('saved ->', OUT, flush=True)
