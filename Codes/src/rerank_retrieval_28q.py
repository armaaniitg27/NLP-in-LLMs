import ast
import json
import os
import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
INPUT_FILE = os.path.join(RESULTS_DIR, 'retrieval_results_28q_rerun.csv')
PASSAGE_FILE = os.path.join(DATA_DIR, 'passage_metadata.jsonl')
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'reranking_results_28q.csv')
SUMMARY_FILE = os.path.join(RESULTS_DIR, 'reranking_summary_28q.csv')
COMPARISON_FILE = os.path.join(RESULTS_DIR, 'reranking_baseline_comparison_28q.csv')
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
CANDIDATE_K = 20
FINAL_K = 5
EMBED_BATCH_SIZE = 16
RERANK_BATCH_SIZE = 2
MAX_LENGTH = 384

def parse_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []

def calculate_metrics(supporting_titles, retrieved_titles):
    supporting = set(supporting_titles)
    retrieved = list(retrieved_titles)
    if not supporting:
        return (np.nan, np.nan, np.nan)
    retrieved_set = set(retrieved)
    recall = len(supporting & retrieved_set) / len(supporting)
    complete = int(supporting.issubset(retrieved_set))
    reciprocal_rank = 0.0
    for rank, title in enumerate(retrieved, start=1):
        if title in supporting:
            reciprocal_rank = 1.0 / rank
            break
    return (recall, complete, reciprocal_rank)

def load_passages(path):
    passages = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            title = item.get('title')
            text = item.get('text')
            if title and text:
                passages.append({'title': title, 'text': text})
    if not passages:
        pass
    return passages

def summarize_results(results):
    rows = []
    for variant, group in results.groupby('variant', sort=False):
        rows.append({'variant': variant, 'queries': len(group), 'mean_candidate_recall_at_20': group['candidate_recall_at_20'].mean(), 'complete_candidate_retrieval_at_20': group['candidate_complete_retrieval_at_20'].mean(), 'mean_reranked_recall_at_5': group['reranked_recall_at_5'].mean(), 'complete_reranked_retrieval_at_5': group['reranked_complete_support_retrieval'].mean(), 'mean_reranked_mrr_at_5': group['reranked_mrr_at_5'].mean()})
    rows.append({'variant': 'ALL', 'queries': len(results), 'mean_candidate_recall_at_20': results['candidate_recall_at_20'].mean(), 'complete_candidate_retrieval_at_20': results['candidate_complete_retrieval_at_20'].mean(), 'mean_reranked_recall_at_5': results['reranked_recall_at_5'].mean(), 'complete_reranked_retrieval_at_5': results['reranked_complete_support_retrieval'].mean(), 'mean_reranked_mrr_at_5': results['reranked_mrr_at_5'].mean()})
    return pd.DataFrame(rows)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(INPUT_FILE):
        pass
    if not os.path.exists(PASSAGE_FILE):
        pass
    queries_df = pd.read_csv(INPUT_FILE)
    required_columns = ['question_id', 'variant', 'original_question', 'query_used', 'reference_answer', 'supporting_titles', 'retrieved_titles', 'retrieved_scores']
    missing_columns = [column for column in required_columns if column not in queries_df.columns]
    if missing_columns:
        pass
    if queries_df.empty:
        pass
    pass
    passages = load_passages(PASSAGE_FILE)
    pass
    pass
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pass
    if device == 'cuda':
        pass
        torch.cuda.empty_cache()
    pass
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    passage_texts = [p['text'] for p in passages]
    pass
    embeddings = embedder.encode(passage_texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    pass
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    del embeddings
    pass
    reranker = CrossEncoder(RERANKER_MODEL, device=device, max_length=MAX_LENGTH)
    output_rows = []
    for position in range(len(queries_df)):
        row = queries_df.iloc[position]
        query = str(row['query_used'])
        query_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        dense_scores, candidate_indices = index.search(query_embedding, CANDIDATE_K)
        candidates = []
        for score, idx in zip(dense_scores[0], candidate_indices[0]):
            if idx < 0:
                continue
            passage = passages[int(idx)]
            candidates.append({'title': passage['title'], 'text': passage['text'], 'dense_score': float(score)})
        if not candidates:
            pass
        pairs = [(query, candidate['text']) for candidate in candidates]
        cross_scores = reranker.predict(pairs, batch_size=RERANK_BATCH_SIZE, show_progress_bar=False)
        for candidate, score in zip(candidates, cross_scores):
            candidate['reranker_score'] = float(score)
        ranked = sorted(candidates, key=lambda item: item['reranker_score'], reverse=True)
        supporting_titles = parse_list(row['supporting_titles'])
        candidate_titles = [item['title'] for item in candidates]
        reranked_top5 = ranked[:FINAL_K]
        reranked_titles = [item['title'] for item in reranked_top5]
        candidate_recall, candidate_complete, _ = calculate_metrics(supporting_titles, candidate_titles)
        recall, complete, mrr = calculate_metrics(supporting_titles, reranked_titles)
        output_rows.append({'question_id': row['question_id'], 'variant': row['variant'], 'original_question': row['original_question'], 'query_used': query, 'reference_answer': row['reference_answer'], 'supporting_titles': json.dumps(supporting_titles), 'candidate_titles': json.dumps(candidate_titles), 'candidate_dense_scores': json.dumps([item['dense_score'] for item in candidates]), 'candidate_recall_at_20': candidate_recall, 'candidate_complete_retrieval_at_20': candidate_complete, 'reranked_titles': json.dumps(reranked_titles), 'reranker_scores': json.dumps([item['reranker_score'] for item in reranked_top5]), 'reranked_passages': json.dumps([{'title': item['title'], 'dense_score': item['dense_score'], 'reranker_score': item['reranker_score'], 'text': item['text']} for item in reranked_top5], ensure_ascii=False), 'reranked_recall_at_5': recall, 'reranked_complete_support_retrieval': complete, 'reranked_mrr_at_5': mrr})
        if device == 'cuda':
            torch.cuda.empty_cache()
        if (position + 1) % 10 == 0 or position + 1 == len(queries_df):
            pass
    results = pd.DataFrame(output_rows)
    results.to_csv(OUTPUT_FILE, index=False)
    summary = summarize_results(results)
    summary.to_csv(SUMMARY_FILE, index=False)
    pass
    pass
    pass
    baseline_rows = []
    for _, row in queries_df.iterrows():
        original_titles = parse_list(row['retrieved_titles'])
        supporting_titles = parse_list(row['supporting_titles'])
        recall, complete, mrr = calculate_metrics(supporting_titles, original_titles)
        baseline_rows.append({'question_id': row['question_id'], 'variant': row['variant'], 'baseline_recall_at_5': recall, 'baseline_complete_retrieval_at_5': complete, 'baseline_mrr_at_5': mrr})
    baseline = pd.DataFrame(baseline_rows)
    comparison = results.merge(baseline, on=['question_id', 'variant'], how='left', validate='one_to_one')
    comparison.to_csv(COMPARISON_FILE, index=False)
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
if __name__ == '__main__':
    main()
