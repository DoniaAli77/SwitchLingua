import pandas as pd

df = pd.read_excel("./output/Arabic_analysis.xlsx")
cols = [
    'topic','cs_ratio','cs_ratio_computed','cs_ratio_score','cs_ratio_notes','sentence_index','refine_count','text'
]
print('Showing first 12 rows (notes truncated to 200 chars):\n')
for i, row in df[cols].head(12).iterrows():
    notes = row['cs_ratio_notes']
    if not isinstance(notes, str):
        notes = str(notes)
    notes = notes.replace('\n',' ').strip()
    text = row['text']
    if isinstance(text, str):
        text = (text[:120] + '...') if len(text) > 120 else text
    print(f"{i+1}. Topic={row['topic']}, target={row['cs_ratio']}, computed={row['cs_ratio_computed']}, score={row['cs_ratio_score']:.2f}, refine_count={row['refine_count']}, sentence_index={row['sentence_index']}")
    print('   Notes:', notes[:200])
    print('   Text:', text)
    print()
