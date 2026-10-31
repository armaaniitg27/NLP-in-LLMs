import csv
import json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
NUM_QUESTIONS = 100
TOP_K = 5
pass
index = faiss.read_index(str(DATA_DIR / 'passages.index'))
with (DATA_DIR / 'passage_metadata.jsonl').open('r', encoding='utf-8') as f:
    passages = [json.loads(line) for line in f]
with (DATA_DIR / 'hotpotqa_sample.jsonl').open('r', encoding='utf-8') as f:
    examples = [json.loads(line) for _, line in zip(range(NUM_QUESTIONS), f)]
pass
pass
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
questions = [example['question'] for example in examples]
pass
embeddings = model.encode(questions, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
pass
scores, indices = index.search(embeddings, TOP_K)
rows = []
recall_values = []
complete_hits = []
mrr_values = []
for i, example in enumerate(examples):
    supporting_titles = set(example['supporting_facts']['title'])
    retrieved_indices = [int(x) for x in indices[i]]
    retrieved_titles = [passages[idx]['title'] for idx in retrieved_indices]
    hits = set(retrieved_titles) & supporting_titles
    recall = len(hits) / len(supporting_titles) if supporting_titles else 0.0
    complete = int(supporting_titles.issubset(set(retrieved_titles)))
    reciprocal_rank = 0.0
    for rank, title in enumerate(retrieved_titles, start=1):
        if title in supporting_titles:
            reciprocal_rank = 1.0 / rank
            break
    recall_values.append(recall)
    complete_hits.append(complete)
    mrr_values.append(reciprocal_rank)
    rows.append({'id': example['id'], 'question': example['question'], 'answer': example['answer'], 'supporting_titles': ' || '.join(sorted(supporting_titles)), 'retrieved_titles': ' || '.join(retrieved_titles), 'supporting_title_recall_at_5': recall, 'complete_title_retrieval_at_5': complete, 'reciprocal_rank_at_5': reciprocal_rank})
output_path = DATA_DIR / 'baseline_retrieval_results.csv'
with output_path.open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
pass
pass
pass
pass
pass
pass
