"""Targeted stronger-model diagnostic on the 18 Design-G Ahmed escalated failures.

Re-runs ONLY the cases G got wrong (final != true) through the full Design-G pipeline
(Lexical + Polarity + Contextual + IntentGate, semantic_v1) on the Ahmed frozen primary,
for two models: gpt-4o-mini (control, same as the original run) and gpt-4.1-mini (stronger).
Question: does the stronger model fix the compliance/knowledge failures, or is any change
just temp-0 noise? Temp 0. Cheap (~18 x 4 x 2 calls).
"""
from __future__ import annotations
import os, sys, json, glob, re, time

sys.path.insert(0, os.path.abspath('.'))
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
TABLE = 'experiments/outputs/multi_agent_bert/experiment_ahmed_designG_intent_gate/error_attribution/attribution_table.json'
OUT = 'experiments/outputs/multi_agent_bert/experiment_G_ahmed_stronger_model'
os.makedirs(OUT, exist_ok=True)
ACTIVE = ['lexical', 'logic', 'contextual', 'polarity']  # polarity slot = IntentGate
MODELS = ['gpt-4o-mini', 'gpt-4.1-mini']
_BLOCK = re.compile(r"intent_gate=BLOCKED override '([^']+)'->'([^']+)'")

tab = json.load(open(TABLE, encoding='utf-8'))
fails = [r for r in tab if r['final_label'] != r['true_label']]
print('Design-G Ahmed failures to retest:', len(fails), flush=True)

bundle = load_task_bundle('src/config/default.yaml', active_task='sentiment_classification',
                          pipeline_mode='full_agentic', threshold=0.7)
tc = bundle.task_config
primary = build_primary_classifier('precomputed', precomputed_predictions=D + '/ahmed_eesa_test_predictions_aligned.csv')


def lab(st, slot):
    o = getattr(st, f'{slot}_output')
    return o.model_output.label if o else None


def run_model(model):
    llm = build_llm_client('openai', llm_model=model, allowed_labels=tc.labels)
    orch = build_orchestrator(tc, threshold=0.7, enable_deliberation=False,
                              keyword_map=bundle.keyword_map, rule_map=bundle.rule_map,
                              primary_classifier=primary, llm_client=llm,
                              consensus_primary_weight=1.0,
                              sentiment_agent_variant='lexical_polarity_contextual_intent_gate')
    rows = []
    for r in fails:
        sid, text, true = r['sample_id'], None, r['true_label']
        # recover full text from the aligned predictions dataset
        text = r['text']
        st = None
        for _ in range(6):
            st = PipelineState(metadata=StateMetadata(sample_id=sid), input_text=text, task_config=tc)
            st = orch.run(st)
            if st.final_output and st.final_output.label is not None and all(lab(st, s) is not None for s in ACTIVE):
                break
            time.sleep(10)
        rationale = st.consensus_output.rationale if st.consensus_output else ''
        m = _BLOCK.search(rationale or '')
        rows.append(dict(sample_id=sid, true_label=true,
                         lexical=lab(st, 'lexical'), polarity=lab(st, 'logic'),
                         contextual=lab(st, 'contextual'), gate=lab(st, 'polarity'),
                         gate_blocked=bool(m), final=st.final_output.label,
                         correct=st.final_output.label == true))
        time.sleep(0.4)
    return rows


results = {}
for mdl in MODELS:
    print('\n=== running', mdl, '===', flush=True)
    results[mdl] = run_model(mdl)
    n_ok = sum(1 for r in results[mdl] if r['correct'])
    print('  %s: now-correct %d / %d of the previously-failing cases' % (mdl, n_ok, len(fails)), flush=True)

json.dump(results, open(OUT + '/diagnostic.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# side-by-side
print('\n=== SIDE BY SIDE (previously all WRONG under original 4o-mini G run) ===', flush=True)
base = {r['sample_id']: r for r in results['gpt-4o-mini']}
strong = {r['sample_id']: r for r in results['gpt-4.1-mini']}
hdr = '%-18s %-9s | %-6s %-6s | %s' % ('sample', 'true', '4o-min', '4.1-mi', 'fixed_by_4.1?')
print(hdr)
fixed = 0
for r in fails:
    sid = r['sample_id']
    b = base[sid]; s = strong[sid]
    only_strong = (s['correct'] and not b['correct'])
    fixed += 1 if only_strong else 0
    print('%-18s %-9s | %-6s %-6s | %s' % (
        sid, r['true_label'],
        'OK' if b['correct'] else b['final'][:5],
        'OK' if s['correct'] else s['final'][:5],
        'YES' if only_strong else ('(both ok)' if b['correct'] and s['correct'] else '')))
print('\nSUMMARY  (of %d previously-failing):' % len(fails), flush=True)
print('  4o-mini re-run now-correct (temp-0 noise floor):', sum(1 for r in results['gpt-4o-mini'] if r['correct']))
print('  4.1-mini now-correct                            :', sum(1 for r in results['gpt-4.1-mini'] if r['correct']))
print('  fixed by 4.1-mini but NOT by 4o-mini re-run     :', fixed)
print('saved ->', OUT, flush=True)
