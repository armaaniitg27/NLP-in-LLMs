import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / 'data' / 'answer_generation_simple_prompt.csv'
OUTPUT_FILE = ROOT / 'data' / 'manual_annotation_dataset.csv'
df = pd.read_csv(INPUT_FILE)
required_columns = ['question_id', 'variant', 'question', 'query_used', 'reference_answer', 'retrieved_titles', 'retrieved_context', 'generated_answer']
missing = [column for column in required_columns if column not in df.columns]
if missing:
    raise ValueError()
df = df.copy()
annotation_columns = {'manual_correctness_assessment': '', 'manual_evidence_support_assessment': '', 'manual_abstention_assessment': '', 'manual_failure_category': '', 'manual_audit_notes': ''}
for column, default_value in annotation_columns.items():
    if column not in df.columns:
        df[column] = default_value
base_columns = [column for column in df.columns if column not in annotation_columns]
df = df[base_columns + list(annotation_columns.keys())]
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
pass
pass
pass
pass
pass
pass
pass
for column in annotation_columns:
    pass
