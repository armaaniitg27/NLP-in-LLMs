import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
PASSAGES_FILE = DATA / 'passage_metadata.jsonl'
OLD_RESULTS_FILE = DATA / 'full_answer_generation_results_0.5B.csv'
PARAPHRASES_FILE = DATA / 'curated_28_question_paraphrases_reconciled.csv'
RESULTS = DATA / "results"
RESULTS.mkdir(exist_ok=True)
OUT_RESULTS = RESULTS / "retrieval_results_28q_rerun.csv"
OUT_METRICS = RESULTS / "retrieval_metrics_28q_rerun.csv"
OUT_SUMMARY = RESULTS / "retrieval_metrics_28q_rerun_summary.csv"
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K = 5
BATCH_SIZE = 64
EXCLUDE_IDS = {'5ac0f7ac554299012d1db666', '5ac3e8c65542997ea680c993'}

def load_jsonl(path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    pass
    return rows

def json_dump(value):
    return json.dumps(value, ensure_ascii=False)

def main():
    for path in (PASSAGES_FILE, OLD_RESULTS_FILE, PARAPHRASES_FILE):
        if not path.exists():
            pass
    passages = load_jsonl(PASSAGES_FILE)
    if not passages:
        pass
    passage_titles = [str(p.get('title', '')).strip() for p in passages]
    passage_texts = [(str(p.get('title', '')).strip() + '\n' + str(p.get('text', '')).strip()).strip() for p in passages]
    if any((not text for text in passage_texts)):
        pass
    old = pd.read_csv(OLD_RESULTS_FILE, keep_default_na=False)
    required_old = {'question_id', 'variant', 'original_question', 'reference_answer', 'supporting_titles'}
    missing = required_old - set(old.columns)
    if missing:
        pass
    old['question_id'] = old['question_id'].astype(str)
    old = old[~old['question_id'].isin(EXCLUDE_IDS)].copy()
    originals = old[old['variant'] == 'original'].drop_duplicates('question_id')
    if originals['question_id'].nunique() != 28:
        pass
    para = pd.read_csv(PARAPHRASES_FILE, keep_default_na=False)
    para['question_id'] = para['question_id'].astype(str)
    para = para[~para['question_id'].isin(EXCLUDE_IDS)].copy()
    if len(para) != 84 or para['question_id'].nunique() != 28:
        pass
    tasks = []
    for _, r in originals.iterrows():
        tasks.append({'question_id': str(r['question_id']), 'variant': 'original', 'original_question': str(r['original_question']), 'query_used': str(r['original_question']), 'reference_answer': str(r['reference_answer']), 'supporting_titles': json.loads(r['supporting_titles']) if str(r['supporting_titles']).strip().startswith('[') else []})
    for _, r in para.iterrows():
        orig_match = originals[originals['question_id'] == str(r['question_id'])]
        if len(orig_match) != 1:
            pass
        old_orig = orig_match.iloc[0]
        try:
            support_titles = json.loads(old_orig['supporting_titles'])
        except Exception:
            support_titles = []
        tasks.append({'question_id': str(r['question_id']), 'variant': str(r['variant']), 'original_question': str(r['original_question']), 'query_used': str(r['curated_paraphrase']), 'reference_answer': str(old_orig['reference_answer']), 'supporting_titles': support_titles})
    order = {'original': 0, 'paraphrase_1': 1, 'paraphrase_2': 2, 'paraphrase_3': 3}
    tasks.sort(key=lambda x: (x['question_id'], order.get(x['variant'], 99)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pass
    pass
    pass
    model = SentenceTransformer(MODEL_NAME, device=device)
    pass
    corpus_embeddings = model.encode(passage_texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    corpus_embeddings = np.ascontiguousarray(corpus_embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)
    queries = [t['query_used'] for t in tasks]
    pass
    query_embeddings = model.encode(queries, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    scores, indices = index.search(query_embeddings, TOP_K)
    result_rows = []
    for i, task in enumerate(tasks):
        retrieved_idxs = [int(x) for x in indices[i] if int(x) >= 0]
        retrieved_titles = [passage_titles[j] for j in retrieved_idxs]
        retrieved_passages = [{'title': passage_titles[j], 'score': float(scores[i][rank]), 'text': str(passages[j].get('text', ''))} for rank, j in enumerate(retrieved_idxs)]
        support = task['supporting_titles']
        found = [title for title in support if title in retrieved_titles]
        missing_titles = [title for title in support if title not in retrieved_titles]
        result_rows.append({'question_id': task['question_id'], 'variant': task['variant'], 'original_question': task['original_question'], 'query_used': task['query_used'], 'reference_answer': task['reference_answer'], 'supporting_titles': json_dump(support), 'retrieved_titles': json_dump(retrieved_titles), 'retrieved_scores': json_dump([float(x) for x in scores[i]]), 'missing_titles': json_dump(missing_titles), 'retrieved_passages': json_dump(retrieved_passages), 'recall_at_5': len(found) / len(support) if support else np.nan, 'complete_support_title_retrieval': int(bool(support) and len(found) == len(support)), 'mrr_at_5': 1.0 / min([retrieved_titles.index(t) + 1 for t in found]) if found else 0.0})
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT_RESULTS, index=False, encoding='utf-8-sig')
    originals_by_id = {r['question_id']: set(json.loads(r['retrieved_titles'])[:TOP_K]) for _, r in results[results['variant'] == 'original'].iterrows()}
    overlaps = []
    for _, r in results.iterrows():
        current = set(json.loads(r['retrieved_titles'])[:TOP_K])
        original_set = originals_by_id[r['question_id']]
        overlaps.append(len(current & original_set) / TOP_K)
    results['top5_overlap_vs_original'] = overlaps
    results.to_csv(OUT_RESULTS, index=False, encoding='utf-8-sig')
    summary = results.groupby('variant', as_index=False).agg(n_questions=('question_id', 'nunique'), mean_recall_at_5=('recall_at_5', 'mean'), complete_support_title_retrieval_rate=('complete_support_title_retrieval', 'mean'), mean_mrr_at_5=('mrr_at_5', 'mean'), mean_top5_overlap_vs_original=('top5_overlap_vs_original', 'mean'))
    summary['corpus_passages'] = len(passages)
    summary['top_k'] = TOP_K
    summary['embedding_model'] = MODEL_NAME
    summary['status'] = 'Fresh consistent 28-question retrieval rerun; title-level metrics'
    summary.to_csv(OUT_SUMMARY, index=False, encoding='utf-8-sig')
    metric_cols = ['question_id', 'variant', 'recall_at_5', 'complete_support_title_retrieval', 'mrr_at_5', 'top5_overlap_vs_original']
    results[metric_cols].to_csv(OUT_METRICS, index=False, encoding='utf-8-sig')
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
