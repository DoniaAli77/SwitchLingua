"""Per-agent re-capture of the 5 cases gpt-4.1-mini BROKE (4o-mini got right) in the
Design-G semantic_v1 run. Shows whether the individual agents neutralized (caution /
instruction-compliance) or the gate blocked (architecture). ~20 calls. Temp 0.
"""
from __future__ import annotations
import os, sys, json, csv, re, time

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

BROKEN = ['ahmed-eesa-00041', 'ahmed-eesa-00045', 'ahmed-eesa-00100', 'ahmed-eesa-00127', 'ahmed-eesa-00362']
D = 'data/Sentiment/external/ahmed'
TABLE = 'experiments/outputs/multi_agent_bert/experiment_ahmed_designG_intent_gate/error_attribution/attribution_table.json'
OUT = 'experiments/outputs/multi_agent_bert/experiment_G_ahmed_gpt41mini/broken_why.txt'
ACTIVE = ['lexical', 'logic', 'contextual', 'polarity']
_BLOCK = re.compile(r"intent_gate=BLOCKED override '([^']+)'->'([^']+)'")

t4o = {r['sample_id']: r for r in json.load(open(TABLE, encoding='utf-8'))}
bundle = load_task_bundle('src/config/default.yaml', active_task='sentiment_classification',
                          pipeline_mode='full_agentic', threshold=0.7)
tc = bundle.task_config
primary = build_primary_classifier('precomputed', precomputed_predictions=D + '/ahmed_eesa_test_predictions_aligned.csv')
llm = build_llm_client('openai', llm_model='gpt-4.1-mini', allowed_labels=tc.labels)
orch = build_orchestrator(tc, threshold=0.7, enable_deliberation=False,
                          keyword_map=bundle.keyword_map, rule_map=bundle.rule_map,
                          primary_classifier=primary, llm_client=llm, consensus_primary_weight=1.0,
                          sentiment_agent_variant='lexical_polarity_contextual_intent_gate')


def lab(st, slot):
    o = getattr(st, f'{slot}_output'); return o.model_output.label if o else None


lines = []
for sid in BROKEN:
    r = t4o[sid]; text = r['text']; true = r['true_label']
    st = None
    for _ in range(6):
        st = PipelineState(metadata=StateMetadata(sample_id=sid), input_text=text, task_config=tc)
        st = orch.run(st)
        if st.final_output and st.final_output.label and all(lab(st, s) is not None for s in ACTIVE):
            break
        time.sleep(10)
    m = _BLOCK.search(st.consensus_output.rationale if st.consensus_output else '')
    lines.append('%s  true=%s' % (sid, true))
    lines.append('  text: ' + text[:110].replace(chr(10), ' '))
    lines.append('  4o-mini : lex=%s pol=%s ctx=%s gate=%s -> final=%s' % (
        r['lexical_label'], r['polarity_label'], r['contextual_label'], r['gate_label'], r['final_label']))
    lines.append('  4.1-mini: lex=%s pol=%s ctx=%s gate=%s -> final=%s%s' % (
        lab(st, 'lexical'), lab(st, 'logic'), lab(st, 'contextual'), lab(st, 'polarity'),
        st.final_output.label, '  [GATE-BLOCKED]' if m else ''))
    lines.append('')
    time.sleep(0.4)

open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('saved ->', OUT)
