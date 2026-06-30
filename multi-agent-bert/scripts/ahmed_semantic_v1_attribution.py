"""Per-agent capture for the 84 escalated samples of Ahmed frozen-primary
full_agentic@0.7 under the semantic_v1 sentiment prompts (B1).

evaluate_pipeline persists only the final prediction, so this re-runs ONLY the 84
escalated samples through the SAME pipeline config (frozen Ahmed primary, threshold
0.7, Fix-2 w_primary=1.0, gpt-4o-mini) with SENTIMENT_PROMPT_VARIANT=semantic_v1 and
captures lexical/logic/contextual/consensus outputs. Temperature is 0.0, so this
faithfully reproduces the B1 run's per-agent decisions. No training, no generation.

Used to fill report items 8 (agent agreement) and 10 (artifact/literalism) and to
compare against the default-prompt table in experiment_ahmed_frozen_primary/.
"""
from __future__ import annotations
import os, sys, json, glob, csv, collections, time

# enable the variant BEFORE importing the pipeline/agents
os.environ['SENTIMENT_PROMPT_VARIANT'] = 'semantic_v1'

# env (OpenAI key) from the shared .env
for line in open('../Modified_Version/.env', encoding='utf-8'):
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    k, v = line.split('=', 1)
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from src.config.loader import load_task_bundle
from src.state.schema import PipelineState, StateMetadata
from src.prompts._sentiment_variant import active_variant
from evaluate_pipeline import build_primary_classifier, build_llm_client, build_orchestrator

assert active_variant() == 'semantic_v1', 'variant gate not active!'
print('sentiment_prompt_variant =', active_variant(), flush=True)

D = 'data/Sentiment/external/ahmed'
RUN = 'experiments/outputs/multi_agent_bert/experiment_ahmed_semantic_v1'
OUT = RUN + '/error_attribution'
os.makedirs(OUT, exist_ok=True)

# escalated 84 from the semantic_v1 th0.7 predictions
pr = json.load(open(glob.glob(RUN + '/full_agentic_th07_semantic_v1/*predictions*.json')[0], encoding='utf-8'))
esc = [x for x in pr if str(x.get('escalated')).lower() == 'true' or x.get('escalated') is True]
print('escalated samples to re-run:', len(esc), flush=True)

bundle = load_task_bundle('src/config/default.yaml', active_task='sentiment_classification',
                          pipeline_mode='full_agentic', threshold=0.7)
tc = bundle.task_config
primary = build_primary_classifier('precomputed', precomputed_predictions=D + '/ahmed_eesa_test_predictions_aligned.csv')
llm = build_llm_client('openai', llm_model='gpt-4o-mini', allowed_labels=tc.labels)
orch = build_orchestrator(tc, threshold=0.7, enable_deliberation=False,
                          keyword_map=bundle.keyword_map, rule_map=bundle.rule_map,
                          primary_classifier=primary, llm_client=llm, consensus_primary_weight=1.0)

def ao(x):
    if x is None: return (None, None)
    mo = x.model_output
    return (mo.label, round(mo.confidence, 4) if mo.confidence is not None else None)

rows = []
for i, s in enumerate(esc):
    sid = s['sample_id']; text = s['input_text']; true = s['true_label']
    st = None
    for attempt in range(8):
        st = PipelineState(metadata=StateMetadata(sample_id=sid), input_text=text, task_config=tc)
        st = orch.run(st)
        ok = (st.final_output and st.final_output.label is not None
              and st.lexical_output and st.lexical_output.model_output.label is not None
              and st.logic_output and st.logic_output.model_output.label is not None
              and st.contextual_output and st.contextual_output.model_output.label is not None)
        if ok:
            break
        time.sleep(20)
    pm = st.primary_model
    lex = ao(st.lexical_output); log_ = ao(st.logic_output); ctx = ao(st.contextual_output)
    cons = st.consensus_output
    final = st.final_output
    a_lbl = pm.label; f_lbl = final.label if final else None
    a_ok = a_lbl == true; f_ok = f_lbl == true
    trans = ('correct_to_correct' if a_ok and f_ok else 'correct_to_wrong' if a_ok and not f_ok
             else 'wrong_to_correct' if not a_ok and f_ok else 'wrong_to_wrong')
    agents = {'lexical': lex[0], 'logic': log_[0], 'contextual': ctx[0]}
    agree = [k for k, v in agents.items() if v == a_lbl]
    disagree = [k for k, v in agents.items() if v is not None and v != a_lbl]
    rows.append(dict(
        sample_id=sid, text=text.replace('\n', ' ')[:120], true_label=true,
        ahmed_label=a_lbl, ahmed_conf=round(pm.confidence, 4),
        ahmed_probs=json.dumps({k: round(v, 4) for k, v in pm.probabilities.items()}, ensure_ascii=False),
        lexical_label=lex[0], lexical_conf=lex[1], logic_label=log_[0], logic_conf=log_[1],
        contextual_label=ctx[0], contextual_conf=ctx[1],
        consensus_label=cons.label if cons else None, final_label=f_lbl,
        ahmed_correct=a_ok, final_correct=f_ok, transition=trans,
        consensus_changed_ahmed=(f_lbl != a_lbl), agents_agree=','.join(agree), agents_disagree=','.join(disagree),
    ))
    if (i + 1) % 10 == 0: print('  done', i + 1, '/', len(esc), flush=True)
    time.sleep(1.0)

keys = list(rows[0].keys())
with open(OUT + '/attribution_table.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
json.dump(rows, open(OUT + '/attribution_table.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# ---- summaries ----
n = len(rows)
ahmed_ok = [r for r in rows if r['ahmed_correct']]
final_acc = round(sum(1 for r in rows if r['final_correct']) / n, 4)
trans_counts = collections.Counter(r['transition'] for r in rows)
print('\n=== SUMMARY 1 ===')
print('escalated n=%d | Ahmed correct=%d wrong=%d | Ahmed-esc acc=%.4f | final-esc acc=%.4f'
      % (n, len(ahmed_ok), n - len(ahmed_ok), round(len(ahmed_ok)/n,4), final_acc))
print('transitions:', dict(trans_counts))

def agent_acc(field):
    valid = [r for r in rows if r[field] is not None]
    return (round(sum(1 for r in valid if r[field] == r['true_label']) / len(valid), 4), len(valid))
print('\n=== SUMMARY 4: agent accuracy on escalated subset ===')
for name, fld in [('ahmed', 'ahmed_label'), ('lexical', 'lexical_label'), ('logic', 'logic_label'),
                  ('contextual', 'contextual_label'), ('final_consensus', 'final_label')]:
    a, cnt = agent_acc(fld); print('  %-16s acc=%.4f (n=%d)' % (name, a, cnt))

# agreement (item 8)
all_agree = [r for r in rows if r['lexical_label'] == r['logic_label'] == r['contextual_label'] and r['lexical_label'] is not None]
def pair(f1, f2):
    v = [r for r in rows if r[f1] is not None and r[f2] is not None]
    return round(sum(1 for r in v if r[f1] == r[f2]) / len(v), 4)
print('\n=== SUMMARY 5: agent agreement (item 8) ===')
print('  all 3 agents agree: %d/%d = %.4f' % (len(all_agree), n, len(all_agree)/n))
print('  pairwise: lex-logic=%.4f lex-ctx=%.4f logic-ctx=%.4f'
      % (pair('lexical_label','logic_label'), pair('lexical_label','contextual_label'), pair('logic_label','contextual_label')))

cw = [r for r in rows if r['transition'] == 'correct_to_wrong']
wc = [r for r in rows if r['transition'] == 'wrong_to_correct']
print('\n=== correct->wrong (%d) | wrong->correct (%d) ===' % (len(cw), len(wc)))
print('  CW pairs (true->final):', dict(collections.Counter('%s->%s' % (r['true_label'], r['final_label']) for r in cw)))
print('\nsaved ->', OUT + '/attribution_table.csv')
