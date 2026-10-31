from pathlib import Path
import pandas as pd
root = Path(__file__).resolve().parent.parent
input_file = root / 'data' / 'full_answer_generation_results_0.5B_filtered28_provisional.csv'
output_file = root / 'data' / 'hallucination_annotation_template_112.csv'
df = pd.read_csv(input_file)
if len(df) != 112:
    raise ValueError()
new_columns = {'hallucination_present': '', 'hallucination_type': '', 'hallucinated_spans': '', 'hallucination_notes': '', 'hallucination_annotation_status': 'pending'}
for column, default in new_columns.items():
    if column not in df.columns:
        df[column] = default
df = df.sort_values(['question_id', 'variant'], kind='stable').reset_index(drop=True)
df.to_csv(output_file, index=False, encoding='utf-8-sig')
pass
pass
pass
pass
pass
pass
pass
