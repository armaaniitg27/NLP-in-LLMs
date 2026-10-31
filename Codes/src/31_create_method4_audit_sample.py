from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / 'data' / 'answer_verification_revision_28q_v2.csv'
OUTPUT_FILE = PROJECT_ROOT / 'data' / 'method4_manual_audit_sample.csv'
df = pd.read_csv(INPUT_FILE).fillna('')
required_columns = ['question_id', 'variant', 'original_question', 'reference_answer', 'initial_answer', 'retrieved_context', 'verification_assessment', 'verification_reason', 'evidence_quotes', 'revision_action', 'revised_answer', 'revision_reason', 'verification_raw_output', 'revision_raw_output', 'verification_error', 'revision_error']
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError()
selected_indices = set()
for keyword in ['Drug in Me Is You', 'largest private hospital']:
    matches = df[df['original_question'].str.contains(keyword, case=False, regex=False)]
    selected_indices.update(matches.index.tolist()[:3])
for assessment in df['verification_assessment'].unique():
    matches = df[df['verification_assessment'] == assessment]
    if not matches.empty:
        selected_indices.add(matches.index[0])
for action in df['revision_action'].unique():
    matches = df[df['revision_action'] == action]
    if not matches.empty:
        selected_indices.add(matches.index[0])
remaining = df.loc[~df.index.isin(selected_indices)]
extra_count = min(6, len(remaining))
if extra_count:
    random_indices = remaining.sample(n=extra_count, random_state=42).index
    selected_indices.update(random_indices)
audit = df.loc[sorted(selected_indices), required_columns].copy()
audit['_priority'] = 1
audit.loc[audit['original_question'].str.contains('Drug in Me Is You|largest private hospital', case=False, regex=True), '_priority'] = 0
audit = audit.sort_values(['_priority', 'question_id', 'variant']).drop(columns='_priority').reset_index(drop=True)
audit.insert(0, 'manual_correctness', '')
audit.insert(1, 'manual_evidence_support', '')
audit.insert(2, 'manual_hallucination', '')
audit.insert(3, 'manual_notes', '')
audit.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
pass
pass
pass
pass
pass
pass
pass
