from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / 'data' / 'hallucination_annotation_pilot_8.csv'
OUTPUT_FILE = ROOT / 'data' / 'hallucination_annotation_pilot_8_rubric_v2.csv'
df = pd.read_csv(INPUT_FILE)
if 'answer_correctness' not in df.columns:
    df['answer_correctness'] = ''
annotation_columns = ['hallucination_present', 'evident_conflict', 'baseless_information', 'hallucinated_spans', 'annotation_notes']
for column in annotation_columns:
    if column not in df.columns:
        df[column] = ''
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
pass
pass
pass
pass
pass
