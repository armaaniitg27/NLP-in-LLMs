from pathlib import Path
import pandas as pd
PROJECT_DIR = Path('D:\\Sem 7\\NLP with LLMs\\rag-prompt-robustness')
DATA_DIR = PROJECT_DIR / 'data'
INPUT_FILE = DATA_DIR / 'RAG_28q_assistant_audit_labels_validated_batch6.csv'
REVIEW_FILE = DATA_DIR / 'validation_sample_blinded.csv'
KEY_FILE = DATA_DIR / 'validation_sample_answer_key.csv'
N_QUESTION_GROUPS = 8
RANDOM_STATE = 42
LABEL_COLUMNS = ['answer_correctness', 'evidence_support', 'abstention', 'failure_type']
REVIEW_COLUMNS = ['question_id', 'variant', 'original_question', 'query_used', 'reference_answer', 'supporting_titles', 'retrieved_titles', 'missing_titles', 'retrieved_context', 'generated_answer']

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError()
    df = pd.read_csv(INPUT_FILE)
    required = REVIEW_COLUMNS + LABEL_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError()
    variant_counts = df.groupby('question_id')['variant'].nunique()
    eligible_ids = variant_counts[variant_counts == 4].index.tolist()
    if len(eligible_ids) < N_QUESTION_GROUPS:
        raise ValueError()
    rng = RANDOM_STATE
    remaining = set(eligible_ids)
    selected_ids = []
    category_sets = {}
    for qid in eligible_ids:
        group = df[df['question_id'] == qid]
        categories = set()
        for col in LABEL_COLUMNS:
            for value in group[col].dropna().astype(str):
                categories.add(f'{col}={value}')
        category_sets[qid] = categories
    covered_categories = set()
    while len(selected_ids) < N_QUESTION_GROUPS:
        candidates = list(remaining)
        candidates = sorted(candidates)
        tie_break = pd.Series(candidates).sample(frac=1, random_state=rng).tolist()
        best_id = max(tie_break, key=lambda qid: len(category_sets[qid] - covered_categories))
        selected_ids.append(best_id)
        covered_categories.update(category_sets[best_id])
        remaining.remove(best_id)
        rng += 1
    sample = df[df['question_id'].isin(selected_ids)].copy()
    key_columns = ['question_id', 'variant', *LABEL_COLUMNS, 'audit_notes', 'review_status']
    answer_key = sample[key_columns].copy()
    review = sample[REVIEW_COLUMNS].copy()
    review['human_correctness'] = ''
    review['human_evidence_support'] = ''
    review['human_abstention'] = ''
    review['human_failure_type'] = ''
    review['human_notes'] = ''
    review = review.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(REVIEW_FILE, index=False)
    answer_key.to_csv(KEY_FILE, index=False)
    pass
    pass
    pass
    pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
