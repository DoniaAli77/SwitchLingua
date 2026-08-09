"""Freeze the 1,044-row silver primary subset and verify it does not overlap the
Topic-540 training data or the ArEnTC splits.

No training, no annotation. Read-only over the corpora; writes one frozen JSONL.

Overlap is checked on NORMALISED text: NFKC, lowercased, Arabic diacritics and
tatweel stripped, Arabic-Indic digits folded to ASCII, punctuation removed, and
whitespace collapsed. Both exact normalised equality and a token-set containment
check are reported so near-duplicates are not missed.
"""
from __future__ import annotations
import csv, json, os, re, sys, unicodedata, collections

sys.path.insert(0, os.path.abspath('.'))

BASE = 'data/Topic'
SILVER = f'{BASE}/transcription data/silver_corpus.csv'
GEN540 = f'{BASE}/generated/merged/switchlingua_topic_train_540_60perlabel.jsonl'
SPLITS = {
    'ARENTCV2/train': f'{BASE}/processed/ARENTCV2/train.jsonl',
    'ARENTCV2/dev':   f'{BASE}/processed/ARENTCV2/dev.jsonl',
    'ARENTCV2/test':  f'{BASE}/processed/ARENTCV2/test.jsonl',
    'ARENTCV1/train': f'{BASE}/processed/ARENTCV1/train.jsonl',
    'ARENTCV1/dev':   f'{BASE}/processed/ARENTCV1/dev.jsonl',
    'ARENTCV1/test':  f'{BASE}/processed/ARENTCV1/test.jsonl',
}
OUT_DIR = 'experiments/outputs/multi_agent_bert/experiment_silver_topic540'
os.makedirs(OUT_DIR, exist_ok=True)
FROZEN = f'{OUT_DIR}/silver_primary_1044.jsonl'

# Corpus label -> checkpoint label. The Topic-540 checkpoints were trained with
# "tech"; the silver corpus uses "technology". Everything else is identical.
LABEL_ALIAS = {'technology': 'tech'}

_DIACRITICS = re.compile(r'[ً-ٰٟـ]')          # harakat + tatweel
_PUNCT = re.compile(r'[^\w\s]', re.UNICODE)
_AR_DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')


def norm(t: str) -> str:
    t = unicodedata.normalize('NFKC', t or '')
    t = t.translate(_AR_DIGITS)
    t = _DIACRITICS.sub('', t)
    t = t.lower()
    t = _PUNCT.sub(' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def load_jsonl_texts(path):
    if not os.path.exists(path):
        return None
    out = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                out.append(d.get('text', ''))
    return out


# ---------------------------------------------------------------- freeze subset
rows = list(csv.DictReader(open(SILVER, encoding='utf-8')))
sid_key = [k for k in rows[0] if k.endswith('segment_id')][0]  # tolerate BOM
sub = [r for r in rows
       if r['cs_verified'] == 'yes'
       and r['cs_category'] == 'lexical_or_phrase'
       and r['standalone'] == 'yes'
       and r['route'] == 'accept_silver_primary']

frozen = []
for r in sub:
    silver = r['topic'].strip().lower()
    frozen.append({
        'segment_id': r[sid_key],
        'text': r['sentence'],
        'label': LABEL_ALIAS.get(silver, silver),   # checkpoint label space
        'silver_topic': silver,                      # original corpus label
        'cs_type': r['cs_type'],
        'video_id': r['video_id'],
        'num_tokens': r['num_tokens'],
        'cmi': r['cmi'],
        'ar_pct': r['ar_pct'],
        'en_pct': r['en_pct'],
    })
with open(FROZEN, 'w', encoding='utf-8') as fh:
    for d in frozen:
        fh.write(json.dumps(d, ensure_ascii=False) + '\n')

print(f'FROZEN SUBSET: {len(frozen)} rows -> {FROZEN}')
print('  label distribution (checkpoint space):',
      dict(sorted(collections.Counter(d['label'] for d in frozen).items())))
print('  cs_type:', dict(collections.Counter(d['cs_type'] for d in frozen)))
print('  unique videos:', len({d['video_id'] for d in frozen}))

# ------------------------------------------------------------- overlap checking
silver_norm = {norm(d['text']): d['segment_id'] for d in frozen}
silver_tok = {sid: set(n.split()) for n, sid in silver_norm.items()}
print(f'\n  unique normalised silver texts: {len(silver_norm)} '
      f'(of {len(frozen)} rows -> {len(frozen)-len(silver_norm)} internal duplicates)')

report = {'frozen_rows': len(frozen), 'unique_normalised': len(silver_norm), 'sources': {}}
print('\nOVERLAP CHECK (normalised exact match + token-set containment)')
print(f'{"source":<22} | {"rows":>7} | {"exact":>6} | {"contained":>9}')
print('-' * 54)

sources = {'Topic-540 train (generated)': GEN540}
sources.update(SPLITS)

for name, path in sources.items():
    texts = load_jsonl_texts(path)
    if texts is None:
        print(f'{name:<22} |  MISSING (file not found)')
        report['sources'][name] = {'status': 'missing'}
        continue
    other_norm = {norm(t) for t in texts}
    exact = set(silver_norm) & other_norm
    # containment: a silver sentence whose token set is a subset of some source
    # sentence (or vice versa) - catches truncation / padding near-duplicates.
    other_tok = [set(n.split()) for n in other_norm if n]
    contained = 0
    if len(silver_tok) and len(other_tok) < 200000:
        big = collections.defaultdict(list)
        for ts in other_tok:
            for tok in ts:
                big[tok].append(ts)
        for sid, ts in silver_tok.items():
            if not ts:
                continue
            rare = min(ts, key=lambda t: len(big.get(t, ())))
            for cand in big.get(rare, ()):
                if ts <= cand or cand <= ts:
                    contained += 1
                    break
    print(f'{name:<22} | {len(texts):>7} | {len(exact):>6} | {contained:>9}')
    report['sources'][name] = {'rows': len(texts), 'exact_overlap': len(exact),
                               'containment_overlap': contained,
                               'examples': sorted(exact)[:3]}

total_exact = sum(v.get('exact_overlap', 0) for v in report['sources'].values())
total_cont = sum(v.get('containment_overlap', 0) for v in report['sources'].values())
print('-' * 54)
print(f'{"TOTAL":<22} | {"":>7} | {total_exact:>6} | {total_cont:>9}')
print(f'\nCLEAN: {"YES - no overlap detected" if (total_exact==0 and total_cont==0) else "NO - see above"}')

# label-space verification against the ArEnTC splits
v2_labels = set()
t = load_jsonl_texts(SPLITS['ARENTCV2/train'])
if t is not None:
    with open(SPLITS['ARENTCV2/train'], encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                v2_labels.add(json.loads(line)['label'])
silver_labels = {d['label'] for d in frozen}
print('\nLABEL-SPACE VERIFICATION')
print('  ArEnTC train labels :', sorted(v2_labels))
print('  silver labels (mapped):', sorted(silver_labels))
print('  identical:', sorted(v2_labels) == sorted(silver_labels))
print('  alias applied:', LABEL_ALIAS)
report['label_check'] = {'arentc_labels': sorted(v2_labels),
                         'silver_labels': sorted(silver_labels),
                         'identical': sorted(v2_labels) == sorted(silver_labels),
                         'alias': LABEL_ALIAS}

json.dump(report, open(f'{OUT_DIR}/overlap_report.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(f'\nsaved -> {OUT_DIR}/overlap_report.json')
