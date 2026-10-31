from pathlib import Path
import pandas as pd
from scipy.stats import binomtest
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = DATA_DIR / 'statistical_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_FILE = DATA_DIR / 'rag_prompt_robustness_combined_annotations.csv'
if not INPUT_FILE.exists():
    pass
df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()
variant_order = ['original', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3']
required = ['question_id', 'variant']
missing_columns = [c for c in required if c not in df.columns]
if missing_columns:
    pass
df['variant'] = df['variant'].astype(str).str.strip().str.lower().str.replace(' ', '_', regex=False)

def find_column(candidates, description):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    pass
correct_col = find_column(['answer_correctness', 'final_correctness'], 'answer correctness')
support_col = find_column(['evidence_support', 'final_evidence_support'], 'evidence support')
hallucination_col = find_column(['hallucination_present', 'hallucination', 'hallucination_label'], 'hallucination presence')
for col in [correct_col, support_col, hallucination_col]:
    df[col] = df[col].astype('string').str.strip().str.lower()
pass
pass
pass
pass
pass
duplicates = df.duplicated(['question_id', 'variant'], keep=False)
pass
if duplicates.any():
    df.loc[duplicates].sort_values(['question_id', 'variant']).to_csv(OUTPUT_DIR / 'duplicate_rows_to_review.csv', index=False)
    pass
expected = pd.MultiIndex.from_product([df['question_id'].dropna().unique(), variant_order], names=['question_id', 'variant'])
actual = pd.MultiIndex.from_frame(df[['question_id', 'variant']])
missing_pairs = expected.difference(actual)
pass
if len(missing_pairs):
    pd.DataFrame(missing_pairs.tolist(), columns=['question_id', 'variant']).to_csv(OUTPUT_DIR / 'missing_question_variant_pairs.csv', index=False)
pass
pass
pass
hall = df[hallucination_col].replace({'yes': 'yes', 'true': 'yes', '1': 'yes', 'present': 'yes', 'no': 'no', 'false': 'no', '0': 'no', 'absent': 'no'})
df['hallucination_binary'] = hall.map({'yes': 1, 'no': 0})
df['fully_correct'] = df[correct_col].eq('correct').astype('Int64')
df['fully_supported'] = df[support_col].eq('supported').astype('Int64')
df.loc[df[correct_col].isna(), 'fully_correct'] = pd.NA
df.loc[df[support_col].isna(), 'fully_supported'] = pd.NA

def counts_and_percentages(series, metric, variant):
    counts = series.value_counts(dropna=False)
    total = len(series)
    rows = []
    for label, count in counts.items():
        rows.append({'variant': variant, 'metric': metric, 'category': 'missing' if pd.isna(label) else str(label), 'count': int(count), 'percent_of_variant': round(100 * count / total, 2) if total else None, 'variant_n': total})
    return rows
summary_rows = []
for variant in variant_order:
    part = df[df['variant'] == variant]
    for col, metric in [(correct_col, 'answer_correctness'), (support_col, 'evidence_support'), (hallucination_col, 'hallucination_presence')]:
        summary_rows.extend(counts_and_percentages(part[col], metric, variant))
pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / 'descriptive_summary.csv', index=False)

def paired_comparison(data, outcome_col, comparison_name, variant):
    wide = data.pivot(index='question_id', columns='variant', values=outcome_col)
    if 'original' not in wide.columns or variant not in wide.columns:
        return {'comparison': comparison_name, 'variant': variant, 'n_paired': 0, 'original_positive': None, 'variant_positive': None, 'original_rate_pct': None, 'variant_rate_pct': None, 'difference_percentage_points': None, 'improved': None, 'worsened': None, 'unchanged': None, 'exact_mcnemar_p': None, 'status': 'Missing original or comparison variant'}
    pair = wide[['original', variant]].dropna()
    original = pair['original'].astype(int)
    changed = pair[variant].astype(int)
    improved = int(((original == 0) & (changed == 1)).sum())
    worsened = int(((original == 1) & (changed == 0)).sum())
    unchanged = int((original == changed).sum())
    n = len(pair)
    original_positive = int(original.sum())
    variant_positive = int(changed.sum())
    original_rate = 100 * original_positive / n if n else None
    variant_rate = 100 * variant_positive / n if n else None
    discordant = improved + worsened
    p_value = binomtest(k=min(improved, worsened), n=discordant, p=0.5, alternative='two-sided').pvalue if discordant else 1.0
    return {'comparison': comparison_name, 'variant': variant, 'n_paired': n, 'original_positive': original_positive, 'variant_positive': variant_positive, 'original_rate_pct': round(original_rate, 2) if n else None, 'variant_rate_pct': round(variant_rate, 2) if n else None, 'difference_percentage_points': round(variant_rate - original_rate, 2) if n else None, 'improved': improved, 'worsened': worsened, 'unchanged': unchanged, 'exact_mcnemar_p': round(p_value, 6), 'status': 'OK'}
metrics = [('fully_correct', 'Fully correct answer'), ('fully_supported', 'Fully evidence-supported answer'), ('hallucination_binary', 'Hallucination present')]
test_rows = []
for outcome_col, metric_name in metrics:
    for variant in variant_order[1:]:
        result = paired_comparison(df, outcome_col, metric_name, variant)
        test_rows.append(result)
tests = pd.DataFrame(test_rows)
tests.to_csv(OUTPUT_DIR / 'paired_comparison_tests.csv', index=False)
transition_rows = []
for outcome_col, metric_name in metrics:
    wide = df.pivot(index='question_id', columns='variant', values=outcome_col)
    if 'original' not in wide.columns:
        continue
    for variant in variant_order[1:]:
        if variant not in wide.columns:
            continue
        pair = wide[['original', variant]].dropna()
        for qid, row in pair.iterrows():
            old = int(row['original'])
            new = int(row[variant])
            if old == new:
                transition = 'unchanged'
            elif old == 0 and new == 1:
                transition = '0_to_1'
            else:
                transition = '1_to_0'
            transition_rows.append({'question_id': qid, 'metric': metric_name, 'comparison_variant': variant, 'original_value': old, 'variant_value': new, 'transition': transition})
pd.DataFrame(transition_rows).to_csv(OUTPUT_DIR / 'question_level_transitions.csv', index=False)
pass
pass
pass
pass
pass
pass
pass
pass
pass
