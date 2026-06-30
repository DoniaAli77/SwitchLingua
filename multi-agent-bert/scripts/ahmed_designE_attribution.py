"""Per-agent capture for Design E (Lexical + Intent + Polarity + Contextual) on the
Ahmed frozen-primary full_agentic@0.7 escalated subset (semantic_v1 prompts).

Deterministic (temp 0) re-run capturing all four slots:
  lexical_output   -> Lexical
  logic_output     -> Polarity (decider, logic slot)
  contextual_output-> Contextual
  polarity_output  -> Intent   (4th slot)
Used for items 10-13 (Intent accuracy, Intent<->Polarity, Intent<->Contextual,
all-agent agreement). No training, no generation.
"""
from __future__ import annotations
import os, sys, json, glob, csv, collections, time

for line in open('../Modified_Version/.env', encoding='utf-8'):
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    k, v = line.split('=', 1)
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ['SENTIMENT_PROMPT_VARIANT'] = 'semantic_v1'
os.environ['SENTIMENT_AGENT_VARIANT'] = 'lexical_intent_polarity_contextual'

from src.config.loader import load_task_bundle
from src.state.schema import PipelineState, StateMetadata
from evaluate_pipeline import build_primary_classifier, build_llm_client, build_orchestrator

D = 'data/Sentiment/external/ahmed'
RUN = 'experiments/outputs/multi_agent_bert/experiment_ahmed_designE_intent'
OUT = RUN + '/error_attribution'
os.makedirs(OUT, exist_ok=True)
# display names per state slot for E
DISP = {"lexical": "lexical", "logic": "polarity", "contextual": "contextual", "polarity": "intent"}
ACTIVE = ["lexical", "logic", "contextual", "polarity"]

pr = json.load(open(glob.glob(RUN + '/full/*predictions.json')[0], encoding='utf-8'))
esc = [x for x in pr if str(x.get('escalated')).lower() == 'true' or x.get('escalated') is True]
print(f"Design E — {len(esc)} escalated", flush=True)

bundle = load_task_bundle('src/config/default.yaml', active_task='sentiment_classification',
                          pipeline_mode='full_agentic', threshold=0.7)
tc = bundle.task_config
primary = build_primary_classifier('precomputed', precomputed_predictions=D + '/ahmed_eesa_test_predictions_aligned.csv')
llm = build_llm_client('openai', llm_model='gpt-4o-mini', allowed_labels=tc.labels)
orch = build_orchestrator(tc, threshold=0.7, enable_deliberation=False,
                          keyword_map=bundle.keyword_map, rule_map=bundle.rule_map,
                          primary_classifier=primary, llm_client=llm,
                          consensus_primary_weight=1.0, sentiment_agent_variant='lexical_intent_polarity_contextual')

def lab(st, slot):
    o = getattr(st, f"{slot}_output")
    return o.model_output.label if o else None

rows = []
for i, s in enumerate(esc):
    sid = s['sample_id']; text = s['input_text']; true = s['true_label']
    st = None
    for attempt in range(8):
        st = PipelineState(metadata=StateMetadata(sample_id=sid), input_text=text, task_config=tc)
        st = orch.run(st)
        if st.final_output and st.final_output.label is not None and all(lab(st, sl) is not None for sl in ACTIVE):
            break
        time.sleep(15)
    pm = st.primary_model
    f_lbl = st.final_output.label if st.final_output else None
    a_lbl = pm.label; a_ok = a_lbl == true; f_ok = f_lbl == true
    trans = ('correct_to_correct' if a_ok and f_ok else 'correct_to_wrong' if a_ok and not f_ok
             else 'wrong_to_correct' if not a_ok and f_ok else 'wrong_to_wrong')
    rows.append(dict(sample_id=sid, text=text.replace('\n', ' ')[:120], true_label=true,
                     ahmed_label=a_lbl, ahmed_conf=round(pm.confidence, 4),
                     lexical_label=lab(st, 'lexical'), polarity_label=lab(st, 'logic'),
                     contextual_label=lab(st, 'contextual'), intent_label=lab(st, 'polarity'),
                     final_label=f_lbl, ahmed_correct=a_ok, final_correct=f_ok, transition=trans))
    if (i + 1) % 10 == 0: print('  done', i + 1, '/', len(esc), flush=True)
    time.sleep(0.7)

json.dump(rows, open(OUT + '/attribution_table.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
with open(OUT + '/attribution_table.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

n = len(rows)
COLS = {"lexical": "lexical_label", "polarity": "polarity_label", "contextual": "contextual_label", "intent": "intent_label"}
def acc(col):
    v = [r for r in rows if r[col] is not None]
    return round(sum(1 for r in v if r[col] == r['true_label']) / len(v), 4)
print("\n--- Design E per-agent accuracy (escalated) ---")
for disp, col in COLS.items():
    print(f"   {disp:11s} acc={acc(col):.4f}")
print(f"   {'final':11s} acc={round(sum(1 for r in rows if r['final_correct'])/n,4):.4f}   ahmed=0.7500")
def pair(c1, c2):
    v = [r for r in rows if r[c1] and r[c2]]
    return round(sum(1 for r in v if r[c1] == r[c2]) / len(v), 4)
allag = sum(1 for r in rows if len({r[c] for c in COLS.values()}) == 1 and all(r[c] for c in COLS.values()))
print(f"\n   all-4-agree {allag}/{n} = {allag/n:.4f}")
print(f"   intent<->polarity   = {pair('intent_label','polarity_label')}")
print(f"   intent<->contextual = {pair('intent_label','contextual_label')}")
print(f"   intent<->lexical    = {pair('intent_label','lexical_label')}")
print(f"   polarity<->contextual = {pair('polarity_label','contextual_label')}")
print(f"   transitions: {dict(collections.Counter(r['transition'] for r in rows))}")
print("saved ->", OUT)
