import json
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CURATED_FILE = DATA / 'curated_30_question_paraphrases.csv'
QUESTION_FILE = DATA / 'prompt_variants.jsonl'
PASSAGE_FILE = DATA / 'passage_metadata.jsonl'
INDEX_FILE = DATA / 'passages.index'
OUTPUT_FILE = DATA / 'curated_retrieval_results.csv'
SUMMARY_FILE = DATA / 'curated_retrieval_summary.csv'
TOP_K = 5
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]

def normalize_title(title):
    return str(title).strip().casefold()

def main():
    curated = pd.read_csv(CURATED_FILE)
    questions = load_jsonl(QUESTION_FILE)
    passages = load_jsonl(PASSAGE_FILE)
    index = faiss.read_index(str(INDEX_FILE))
    question_lookup = {str(item['id']): item for item in questions}
    if len(curated) != 90:
        raise ValueError()
    if curated['question_id'].nunique() != 30:
        raise ValueError()
    if not curated.groupby('question_id').size().eq(3).all():
        raise ValueError()
    if len(passages) != index.ntotal:
        raise ValueError()
    passage_titles = [item['title'] for item in passages]
    title_to_indices = {}
    for position, title in enumerate(passage_titles):
        title_to_indices.setdefault(normalize_title(title), []).append(position)
    model = SentenceTransformer(MODEL_NAME)
    records = []
    for question_id, group in curated.groupby('question_id', sort=False):
        question_id = str(question_id)
        if question_id not in question_lookup:
            raise ValueError()
        source = question_lookup[question_id]
        supporting_titles = source['supporting_titles']
        supporting_set = {normalize_title(title) for title in supporting_titles}
        original_question = group.iloc[0]['original_question']
        queries = [('original', original_question)]
        for _, row in group.sort_values('candidate_number').iterrows():
            queries.append((f"paraphrase_{int(row['candidate_number'])}", row['curated_paraphrase']))
        query_texts = [query for _, query in queries]
        embeddings = model.encode(query_texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype('float32')
        scores, indices = index.search(embeddings, TOP_K)
        retrieved_titles_by_variant = {}
        for query_index, (variant, query) in enumerate(queries):
            retrieved_indices = [int(i) for i in indices[query_index] if i >= 0]
            retrieved_titles = [passage_titles[i] for i in retrieved_indices]
            retrieved_set = {normalize_title(title) for title in retrieved_titles}
            hits = retrieved_set & supporting_set
            recall = len(hits) / len(supporting_set) if supporting_set else np.nan
            first_relevant_rank = next((rank for rank, title in enumerate(retrieved_titles, start=1) if normalize_title(title) in supporting_set), None)
            reciprocal_rank = 1 / first_relevant_rank if first_relevant_rank is not None else 0.0
            records.append({'question_id': question_id, 'variant': variant, 'query': query, 'supporting_titles': json.dumps(supporting_titles, ensure_ascii=False), 'retrieved_titles': json.dumps(retrieved_titles, ensure_ascii=False), 'recall_at_5': recall, 'complete_support_retrieval': int(supporting_set.issubset(retrieved_set)), 'first_relevant_rank': first_relevant_rank, 'reciprocal_rank_at_5': reciprocal_rank, 'retrieval_scores': json.dumps([float(s) for s in scores[query_index]])})
            retrieved_titles_by_variant[variant] = retrieved_set
        original_set = retrieved_titles_by_variant['original']
        for record in records[-len(queries):]:
            variant = record['variant']
            current_set = retrieved_titles_by_variant[variant]
            record['top5_overlap_with_original'] = len(current_set & original_set) / TOP_K
    results = pd.DataFrame(records)
    results.to_csv(OUTPUT_FILE, index=False)
    summary = results.groupby('variant').agg(questions=('question_id', 'nunique'), mean_recall_at_5=('recall_at_5', 'mean'), complete_retrieval_rate=('complete_support_retrieval', 'mean'), mean_mrr_at_5=('reciprocal_rank_at_5', 'mean'), mean_top5_overlap=('top5_overlap_with_original', 'mean')).reset_index()
    summary.to_csv(SUMMARY_FILE, index=False)
    pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
