"""Sum-rule vs hard-vote fusion on TOPIC (ARENTCV2, XLM-R, gpt-4.1-mini) — the
9-label flip-test. Confirms whether the sum rule flips any label on the harder
9-class task before considering a default migration.

Runs the topic full_agentic pipeline once over the 48-escalated test subset (the
fusion-active samples), capturing each voter's (label, confidence, probabilities)
+ the primary's real 9-way softmax. Re-fuses each sample OFFLINE with the real
ConsensusAgent under both fusion modes. Topic has NO IntentGate (3 voters:
lexical + logic + contextual). No training, no generation.
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

from src.config.loader import load_task_bundle
from src.state.schema import PipelineState, StateMetadata, AgentOutput
from src.agents.consensus_agent import ConsensusAgent
from evaluate_pipeline import build_primary_classifier, build_llm_client, build_orchestrator

MODEL = 'gpt-4.1-mini'
THRESH = 0.90
CKPT = 'experiments/checkpoints/topic_arentcv2_xlmr'
TEST = 'data/Topic/processed/ARENTCV2/test_escalated48.jsonl'
OUTDIR = 'experiments/outputs/multi_agent_bert/experiment_topic_sumrule_vs_hardvote'
os.makedirs(OUTDIR, exist_ok=True)
ACTIVE = ['lexical', 'logic', 'contextual']

lm = json.load(open(CKPT + '/label_map.json', encoding='utf-8'))
label_map = {int(k): v for k, v in lm['id2label'].items()}

bundle = load_task_bundle('src/config/default.yaml', active_task='topic_classification',
                          pipeline_mode='full_agentic', threshold=THRESH)
tc = bundle.task_config
LABELS = tc.labels
primary = build_primary_classifier('transformer', transformer_checkpoint=CKPT,
                                   device='cuda', label_map=label_map)
llm = build_llm_client('openai', llm_model=MODEL, allowed_labels=LABELS)
orch = build_orchestrator(tc, threshold=THRESH, enable_deliberation=False,
                          keyword_map=bundle.keyword_map, rule_map=bundle.rule_map,
                          primary_classifier=primary, llm_client=llm,
                          consensus_primary_weight=1.0)

CONS = {f: ConsensusAgent(fusion=f, weights={'primary': 1.0}) for f in ('hard_vote', 'sum_rule')}

def refuse(fusion, voters, pmo):
    st = PipelineState(metadata=StateMetadata(sample_id='x'), input_text='', task_config=tc)
    st.lexical_output, st.logic_output, st.contextual_output = voters
    st.primary_model_output = pmo
    st = CONS[fusion].run(st)
    return st.final_output.label, float(st.final_output.confidence)

samples = [json.loads(l) for l in open(TEST, encoding='utf-8') if l.strip()]
records = []
for i, s in enumerate(samples):
    sid = s['id']; text = s['text']; true = s['label']
    st = None
    for _ in range(8):
        st = PipelineState(metadata=StateMetadata(sample_id=sid), input_text=text, task_config=tc)
        st = orch.run(st)
        if (st.final_output and st.final_output.label is not None
                and all(getattr(st, f'{sl}_output') and getattr(st, f'{sl}_output').model_output.label
                        for sl in ACTIVE)):
            break
        time.sleep(15)
    pmo = st.primary_model_output
    escalated = st.metadata.escalated if hasattr(st.metadata, 'escalated') else True
    voters = tuple(AgentOutput(agent_name=sl, model_output=getattr(st, f'{sl}_output').model_output)
                   for sl in ACTIVE)
    hl, hc = refuse('hard_vote', voters, pmo)
    sl_, sc = refuse('sum_rule', voters, pmo)
    records.append(dict(sample_id=sid, true=true,
                        primary_label=pmo.label, primary_conf=round(float(pmo.confidence), 6),
                        hard_label=hl, hard_conf=round(hc, 6),
                        sum_label=sl_, sum_conf=round(sc, 6)))
    if (i + 1) % 10 == 0:
        print(f'  {i+1}/{len(samples)} done', flush=True)
    time.sleep(0.5)

json.dump(records, open(OUTDIR + '/refusion_records.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

def acc(lab): return sum(1 for r in records if r[lab] == r['true']) / len(records)
def ece(lab, conf, bins=10):
    n = len(records); tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [r for r in records if (lo < r[conf] <= hi) or (b == 0 and r[conf] == 0.0)]
        if not bucket: continue
        a = sum(1 for r in bucket if r[lab] == r['true']) / len(bucket)
        c = sum(r[conf] for r in bucket) / len(bucket)
        tot += (len(bucket) / n) * abs(a - c)
    return tot
def mean_conf(lab, conf, correct):
    v = [r[conf] for r in records if (r[lab] == r['true']) == correct]
    return sum(v) / len(v) if v else float('nan')
flips = [r for r in records if r['hard_label'] != r['sum_label']]

o = io.StringIO()
def P(s=''): o.write(s + '\n')
P('TOPIC SUM-RULE vs HARD-VOTE — ARENTCV2 XLM-R, %s, threshold %.2f' % (MODEL, THRESH))
P('escalated (fusion-active) samples: %d | labels: %d' % (len(records), len(LABELS)))
P('LABEL FLIPS (hard vs sum): %d' % len(flips))
for r in flips:
    P('   FLIP %s true=%s hard=%s sum=%s' % (r['sample_id'], r['true'], r['hard_label'], r['sum_label']))
P('')
P('%-22s | %-10s | %-10s' % ('metric', 'hard_vote', 'sum_rule'))
P('-' * 50)
P('%-22s | %.4f     | %.4f' % ('accuracy (escalated)', acc('hard_label'), acc('sum_label')))
P('%-22s | %.4f     | %.4f' % ('ECE (lower=better)', ece('hard_label', 'hard_conf'), ece('sum_label', 'sum_conf')))
P('%-22s | %.4f     | %.4f' % ('mean conf | correct', mean_conf('hard_label', 'hard_conf', True), mean_conf('sum_label', 'sum_conf', True)))
P('%-22s | %.4f     | %.4f' % ('mean conf | wrong', mean_conf('hard_label', 'hard_conf', False), mean_conf('sum_label', 'sum_conf', False)))
open(OUTDIR + '/topic_sumrule_vs_hardvote_report.txt', 'w', encoding='utf-8').write(o.getvalue())
print('\n' + o.getvalue())
print('saved ->', OUTDIR, flush=True)
