"""Sum-rule vs hard-vote fusion on Ahmed (G2, gpt-4.1-mini) — ONE paid pass.

Runs the G2 selective-gate pipeline once over the escalated subset (the only
samples that call the LLM), capturing each voter's (label, confidence,
probabilities) + the primary softmax + the gate's decision. Then re-fuses each
sample OFFLINE with the real ConsensusAgent under both fusion modes
("hard_vote" and "sum_rule") and replays the IntentGate, so both final labels
and both reported confidences come from the production code path — no second
LLM call. Non-escalated samples pass through as the primary (identical under
both schemes). Emits accuracy (must match) + calibration (ECE, mean-conf on
correct vs wrong) for the two schemes. No training, no generation.
"""
from __future__ import annotations
import os, sys, json, glob, csv, time, io

sys.path.insert(0, os.path.abspath('.'))
for line in open('../Modified_Version/.env', encoding='utf-8'):
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    k, v = line.split('=', 1)
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ['SENTIMENT_PROMPT_VARIANT'] = 'semantic_v1'
os.environ['SENTIMENT_AGENT_VARIANT'] = 'lexical_polarity_contextual_selective_gate'

from src.config.loader import load_task_bundle
from src.state.schema import PipelineState, StateMetadata, AgentOutput, ModelOutput
from src.agents.consensus_agent import ConsensusAgent
from src.agents.intent_gate_agent import IntentGateAgent
from evaluate_pipeline import build_primary_classifier, build_llm_client, build_orchestrator

MODEL = 'gpt-4.1-mini'
THRESH = 0.7
D = 'data/Sentiment/external/ahmed'
ALIGNED = D + '/ahmed_eesa_test_predictions_aligned.csv'
OUTDIR = 'experiments/outputs/multi_agent_bert/experiment_ahmed_sumrule_vs_hardvote'
os.makedirs(OUTDIR, exist_ok=True)
ACTIVE = ['lexical', 'logic', 'contextual']  # the 3 voters (polarity slot = gate)

bundle = load_task_bundle('src/config/default.yaml', active_task='sentiment_classification',
                          pipeline_mode='full_agentic', threshold=THRESH)
tc = bundle.task_config
LABELS = tc.labels
primary = build_primary_classifier('precomputed', precomputed_predictions=ALIGNED)
llm = build_llm_client('openai', llm_model=MODEL, allowed_labels=LABELS)
orch = build_orchestrator(tc, threshold=THRESH, enable_deliberation=False,
                          keyword_map=bundle.keyword_map, rule_map=bundle.rule_map,
                          primary_classifier=primary, llm_client=llm,
                          consensus_primary_weight=1.0,
                          sentiment_agent_variant='lexical_polarity_contextual_selective_gate')

# All 818 samples + which escalate (primary confidence < threshold).
allrows = list(csv.DictReader(open(ALIGNED, encoding='utf-8')))
def primary_mo(r):
    return ModelOutput(label=r['pred_label'], confidence=float(r['confidence']),
                       probabilities={'positive': float(r['prob_positive']),
                                      'negative': float(r['prob_negative']),
                                      'neutral': float(r['prob_neutral'])})

CONS = {f: ConsensusAgent(fusion=f, weights={'primary': 1.0}) for f in ('hard_vote', 'sum_rule')}
GATE = IntentGateAgent(gate_slot='polarity')

def refuse(fusion, voters, gate_out, pmo):
    """Run production ConsensusAgent(fusion)+IntentGate on captured outputs."""
    st = PipelineState(metadata=StateMetadata(sample_id='x'), input_text='', task_config=tc)
    st.lexical_output, st.logic_output, st.contextual_output = voters
    st.polarity_output = gate_out          # the non-voting gate slot
    st.primary_model_output = pmo
    st = CONS[fusion].run(st)
    st = GATE.run(st)
    return st.final_output.label, float(st.final_output.confidence)

records = []          # one per sample: label+conf under each scheme
esc_seen = 0
for i, r in enumerate(allrows):
    sid = r['sample_id']; true = r['true_label']; pmo = primary_mo(r)
    escalated = pmo.confidence < THRESH
    if not escalated:
        # No fusion: final == primary under both schemes.
        records.append(dict(sample_id=sid, true=true, escalated=False, gate_fired=False,
                            hard_label=pmo.label, hard_conf=round(pmo.confidence, 6),
                            sum_label=pmo.label, sum_conf=round(pmo.confidence, 6)))
        continue
    esc_seen += 1
    st = None
    for _ in range(8):
        st = PipelineState(metadata=StateMetadata(sample_id=sid), input_text=r['text'], task_config=tc)
        st = orch.run(st)
        if (st.final_output and st.final_output.label is not None
                and all(getattr(st, f'{sl}_output') and getattr(st, f'{sl}_output').model_output.label
                        for sl in ACTIVE)):
            break
        time.sleep(15)
    voters = tuple(AgentOutput(agent_name=sl, model_output=getattr(st, f'{sl}_output').model_output)
                   for sl in ACTIVE)
    gate_out = st.polarity_output
    hl, hc = refuse('hard_vote', voters, gate_out, primary_mo(r))
    sl_, sc = refuse('sum_rule', voters, gate_out, primary_mo(r))
    gate_fired = 'intent_gate=BLOCKED' in (st.consensus_output.rationale or '') if st.consensus_output else False
    records.append(dict(sample_id=sid, true=true, escalated=True, gate_fired=gate_fired,
                        hard_label=hl, hard_conf=round(hc, 6),
                        sum_label=sl_, sum_conf=round(sc, 6)))
    if esc_seen % 10 == 0:
        print(f'  escalated {esc_seen} done (sample {i+1}/{len(allrows)})', flush=True)
    time.sleep(0.6)

json.dump(records, open(OUTDIR + '/refusion_records.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

# ---- metrics -------------------------------------------------------------
def acc(recs, lab): return sum(1 for r in recs if r[lab] == r['true']) / len(recs)
def ece(recs, lab, conf, bins=10):
    n = len(recs); tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [r for r in recs if (lo < r[conf] <= hi) or (b == 0 and r[conf] == 0.0)]
        if not bucket: continue
        a = sum(1 for r in bucket if r[lab] == r['true']) / len(bucket)
        c = sum(r[conf] for r in bucket) / len(bucket)
        tot += (len(bucket) / n) * abs(a - c)
    return tot
def mean_conf(recs, lab, conf, correct):
    v = [r[conf] for r in recs if (r[lab] == r['true']) == correct]
    return sum(v) / len(v) if v else float('nan')
def label_flips(recs): return sum(1 for r in recs if r['hard_label'] != r['sum_label'])

esc = [r for r in records if r['escalated']]
o = io.StringIO()
def P(s=''): o.write(s + '\n')
P('SUM-RULE vs HARD-VOTE — Ahmed G2 selective gate, %s, threshold %.2f' % (MODEL, THRESH))
P('samples: %d total, %d escalated (fusion active), %d non-escalated' %
  (len(records), len(esc), len(records) - len(esc)))
P('label flips (hard vs sum, over ALL samples): %d' % label_flips(records))
P('')
P('%-22s | %-10s | %-10s' % ('metric', 'hard_vote', 'sum_rule'))
P('-' * 50)
P('%-22s | %.4f     | %.4f' % ('accuracy (full 818)', acc(records, 'hard_label'), acc(records, 'sum_label')))
P('%-22s | %.4f     | %.4f' % ('accuracy (escalated)', acc(esc, 'hard_label'), acc(esc, 'sum_label')))
P('')
P('CALIBRATION on escalated subset (where fusion sets the confidence):')
P('%-22s | %.4f     | %.4f' % ('ECE (lower=better)', ece(esc, 'hard_label', 'hard_conf'), ece(esc, 'sum_label', 'sum_conf')))
P('%-22s | %.4f     | %.4f' % ('mean conf | correct', mean_conf(esc, 'hard_label', 'hard_conf', True), mean_conf(esc, 'sum_label', 'sum_conf', True)))
P('%-22s | %.4f     | %.4f' % ('mean conf | wrong', mean_conf(esc, 'hard_label', 'hard_conf', False), mean_conf(esc, 'sum_label', 'sum_conf', False)))
P('%-22s | %.4f' % ('escalated accuracy', acc(esc, 'hard_label')))
P('  (ideal mean-conf-correct ~= escalated accuracy; larger correct-minus-wrong gap = better separation)')
P('')
P('CALIBRATION on full 818:')
P('%-22s | %.4f     | %.4f' % ('ECE (lower=better)', ece(records, 'hard_label', 'hard_conf'), ece(records, 'sum_label', 'sum_conf')))
open(OUTDIR + '/sumrule_vs_hardvote_report.txt', 'w', encoding='utf-8').write(o.getvalue())
print('\n' + o.getvalue())
print('saved ->', OUTDIR, flush=True)
