"""Offline simulation of alternative consensus rules on the 84 Ahmed-escalated samples.
Uses the already-captured agent outputs (attribution_table.json). NO LLM calls.
Faithfully replicates src/agents/consensus_agent.py scoring + tie-break.
"""
from __future__ import annotations
import json, collections

LABELS = ['positive', 'negative', 'neutral']
D = 'experiments/outputs/multi_agent_bert/experiment_ahmed_frozen_primary/error_attribution'
rows = json.load(open(D + '/attribution_table.json', encoding='utf-8'))

# Ahmed primary_only: 757/818 correct overall; on the 84 escalated Ahmed got 63.
# Non-escalated (734) Ahmed-correct = 757 - 63 = 694 (those keep the primary label).
N_TEST = 818
NONESC_CORRECT = 694

def consensus(agent_votes, primary_label, primary_conf, w_primary):
    """agent_votes: list of (label, conf, weight). Returns winning label."""
    scores = {l: 0.0 for l in LABELS}
    vote_counts = {l: 0 for l in LABELS}
    max_contrib = {l: 0.0 for l in LABELS}
    active = 0.0
    for label, conf, w in agent_votes:
        if label is None or w == 0 or label not in LABELS or conf is None:
            continue
        contrib = w * conf
        scores[label] += contrib; active += w
        vote_counts[label] += 1
        max_contrib[label] = max(max_contrib[label], contrib)
    if active == 0:                      # all agents abstained -> defer to primary
        return primary_label
    pl = None
    if w_primary > 0 and primary_label in LABELS and primary_conf is not None:
        pl = primary_label
        scores[pl] += w_primary * primary_conf; active += w_primary
    mx = max(scores.values())
    tied = sorted([l for l in LABELS if abs(scores[l] - mx) <= 1e-9])
    if len(tied) == 1:
        return tied[0]
    if pl is not None and pl in tied:    # tie-break 1: primary's label
        return pl
    bc = max(vote_counts[l] for l in tied)           # 2: most voting agents
    cands = [l for l in tied if vote_counts[l] == bc]
    if len(cands) == 1:
        return cands[0]
    bm = max(max_contrib[l] for l in cands)          # 3: highest single contribution
    cands = [l for l in cands if abs(max_contrib[l] - bm) <= 1e-9]
    return cands[0] if len(cands) == 1 else sorted(cands)[0]

def agents_of(r):
    return [(r['lexical_label'], r['lexical_conf']), (r['logic_label'], r['logic_conf']),
            (r['contextual_label'], r['contextual_conf'])]

def majority_label(r):
    labs = [r['lexical_label'], r['logic_label'], r['contextual_label']]
    c = collections.Counter([x for x in labs if x is not None])
    if not c: return None, None
    lab, cnt = c.most_common(1)[0]
    if cnt < 2: return None, None        # 3-way split -> no bloc
    confs = [cf for (l, cf) in agents_of(r) if l == lab and cf is not None]
    return lab, (sum(confs) / len(confs) if confs else None)

def evaluate(rule_fn, name):
    cw = wc = overrides = corr = 0
    broke = collections.Counter(); fixed = collections.Counter()
    for r in rows:
        a = r['ahmed_label']; t = r['true_label']
        final = rule_fn(r)
        if final == t: corr += 1
        if final != a:
            overrides += 1
            if a == t and final != t: cw += 1; broke['%s->%s' % (a, final)] += 1
            if a != t and final == t: wc += 1; fixed['%s->%s' % (a, final)] += 1
    esc_acc = corr / len(rows)
    full_acc = (NONESC_CORRECT + corr) / N_TEST
    return dict(name=name, esc_acc=round(esc_acc, 4), full_acc=round(full_acc, 4),
                wc=wc, cw=cw, net=wc - cw, overrides=overrides,
                broke=dict(broke), fixed=dict(fixed))

# ---- rule definitions ----
def current(r):
    return consensus([(l, c, 1.0) for l, c in agents_of(r)], r['ahmed_label'], r['ahmed_conf'], 1.0)

def wprimary(w):
    return lambda r: consensus([(l, c, 1.0) for l, c in agents_of(r)], r['ahmed_label'], r['ahmed_conf'], w)

def bloc(w):
    def f(r):
        bl, bc = majority_label(r)
        votes = [(bl, bc, 1.0)] if bl is not None else []
        return consensus(votes, r['ahmed_label'], r['ahmed_conf'], w)
    return f

def nologic(w):
    return lambda r: consensus([(r['lexical_label'], r['lexical_conf'], 1.0),
                                (r['contextual_label'], r['contextual_conf'], 1.0)],
                               r['ahmed_label'], r['ahmed_conf'], w)

def conservative(require_conf=None):
    def f(r):
        a = r['ahmed_label']
        labs = [r['lexical_label'], r['logic_label'], r['contextual_label']]
        confs = [r['lexical_conf'], r['logic_conf'], r['contextual_conf']]
        all_agree_against = (labs[0] is not None and labs[0] == labs[1] == labs[2] and labs[0] != a)
        if require_conf is not None:
            all_agree_against = all_agree_against and all((c or 0) >= require_conf for c in confs)
        return labs[0] if all_agree_against else a   # keep Ahmed unless unanimous (and confident)
    return f

def neutral_guard(conf_th=0.8):
    base = wprimary(1.0)
    def f(r):
        a = r['ahmed_label']; w = base(r)
        if a == 'neutral' and w in ('positive', 'negative'):
            labs = [r['lexical_label'], r['logic_label'], r['contextual_label']]
            confs = [r['lexical_conf'], r['logic_conf'], r['contextual_conf']]
            unanimous_polar = (labs[0] is not None and labs[0] == labs[1] == labs[2] == w
                               and all((c or 0) >= conf_th for c in confs))
            return w if unanimous_polar else a       # keep Ahmed's neutral otherwise
        return w
    return f

rules = [
    (current, '1. current (w_p=1, agents 1/1/1)'),
    (wprimary(2.0), '2a. w_primary=2'),
    (wprimary(3.0), '2b. w_primary=3'),
    (wprimary(4.0), '2c. w_primary=4'),
    (bloc(1), '3a. agent-bloc, w_primary=1'),
    (bloc(2), '3b. agent-bloc, w_primary=2'),
    (bloc(3), '3c. agent-bloc, w_primary=3'),
    (nologic(1), '4a. no-logic, w_primary=1'),
    (nologic(2), '4b. no-logic, w_primary=2'),
    (nologic(3), '4c. no-logic, w_primary=3'),
    (conservative(None), '5a. conservative: override only if ALL 3 agree vs Ahmed'),
    (conservative(0.8), '5b. conservative + agent conf>=0.8'),
    (neutral_guard(0.8), '6. neutral guard (keep Ahmed-neutral unless 3 agree polar & conf>=0.8)'),
]

print('Ahmed primary_only (escalated) = 0.7500 | full = 0.9254')
print('%-58s %7s %7s %5s %5s %5s %5s' % ('rule', 'escAcc', 'fullAcc', 'W>C', 'C>W', 'net', 'ovr'))
results = []
for fn, name in rules:
    res = evaluate(fn, name); results.append(res)
    print('%-58s %7.4f %7.4f %5d %5d %+5d %5d' % (name, res['esc_acc'], res['full_acc'], res['wc'], res['cw'], res['net'], res['overrides']))
json.dump(results, open(D + '/consensus_simulation_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nconfusion detail (broke / fixed) per rule:')
for res in results:
    if res['broke'] or res['fixed']:
        print('  %-58s broke=%s fixed=%s' % (res['name'], res['broke'], res['fixed']))
print('\nsaved', D + '/consensus_simulation_results.json')
