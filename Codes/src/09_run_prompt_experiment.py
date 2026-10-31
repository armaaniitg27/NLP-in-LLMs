import json
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
VARIANTS_FILE = DATA_DIR / 'prompt_variants.jsonl'
INDEX_FILE = DATA_DIR / 'passages.index'
METADATA_FILE = DATA_DIR / 'passage_metadata.jsonl'
RESULTS_FILE = DATA_DIR / 'prompt_retrieval_results.csv'
SUMMARY_FILE = DATA_DIR / 'prompt_retrieval_summary.csv'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K = 5

def load_jsonl(path):
    with path.open('r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def calculate_metrics(retrieved_titles, supporting_titles):
    retrieved_set = set(retrieved_titles)
    supporting_set = set(supporting_titles)
    recall = len(retrieved_set & supporting_set) / len(supporting_set) if supporting_set else 0.0
    complete_retrieval = int(supporting_set.issubset(retrieved_set))
    reciprocal_rank = 0.0
    for rank, title in enumerate(retrieved_titles, start=1):
        if title in supporting_set:
            reciprocal_rank = 1.0 / rank
            break
    return (recall, complete_retrieval, reciprocal_rank)

def main():
    examples = load_jsonl(VARIANTS_FILE)
    metadata = load_jsonl(METADATA_FILE)
    index = faiss.read_index(str(INDEX_FILE))
    model = SentenceTransformer(MODEL_NAME)
    if index.ntotal != len(metadata):
        raise ValueError()
    if index.d != model.get_sentence_embedding_dimension():
        raise ValueError()
    all_results = []
    total_runs = sum((len(item['variants']) for item in examples))
    completed_runs = 0
    for example in examples:
        question_id = example['id']
        supporting_titles = example['supporting_titles']
        original_titles = None
        question_results = []
        for variant_name, prompt in example['variants'].items():
            embedding = model.encode([prompt], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            scores, indices = index.search(embedding, TOP_K)
            retrieved_titles = [metadata[int(idx)]['title'] for idx in indices[0] if idx >= 0]
            recall, complete, reciprocal_rank = calculate_metrics(retrieved_titles, supporting_titles)
            if variant_name == 'original':
                original_titles = retrieved_titles
            result = {'id': question_id, 'variant': variant_name, 'question': example['question'], 'prompt': prompt, 'answer': example['answer'], 'supporting_titles': json.dumps(supporting_titles), 'retrieved_titles': json.dumps(retrieved_titles), 'similarity_scores': json.dumps([float(score) for score in scores[0]]), 'recall_at_5': recall, 'complete_support_retrieval': complete, 'reciprocal_rank_at_5': reciprocal_rank}
            question_results.append(result)
            completed_runs += 1
            if completed_runs % 50 == 0:
                pass
        if original_titles is None:
            raise ValueError()
        original_set = set(original_titles)
        for result in question_results:
            variant_titles = set(json.loads(result['retrieved_titles']))
            result['top5_overlap_with_original'] = len(original_set & variant_titles) / TOP_K
            all_results.append(result)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_FILE, index=False)
    summary_df = results_df.groupby('variant', sort=False).agg(questions=('id', 'nunique'), mean_recall_at_5=('recall_at_5', 'mean'), complete_retrieval_rate=('complete_support_retrieval', 'mean'), mean_mrr_at_5=('reciprocal_rank_at_5', 'mean'), mean_top5_overlap=('top5_overlap_with_original', 'mean')).reset_index()
    summary_df.to_csv(SUMMARY_FILE, index=False)
    pass
    pass
    pass
    pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
