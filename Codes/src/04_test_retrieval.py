import json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
index_path = DATA_DIR / 'passages.index'
metadata_path = DATA_DIR / 'passage_metadata.jsonl'
questions_path = DATA_DIR / 'hotpotqa_sample.jsonl'
pass
index = faiss.read_index(str(index_path))
pass
with metadata_path.open('r', encoding='utf-8') as f:
    passages = [json.loads(line) for line in f]
pass
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
with questions_path.open('r', encoding='utf-8') as f:
    first_example = json.loads(next(f))
question = first_example['question']
answer = first_example['answer']
supporting_titles = set(first_example['supporting_facts']['title'])
pass
pass
pass
query_embedding = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
scores, indices = index.search(query_embedding, 5)
pass
retrieved_titles = []
for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
    passage = passages[int(idx)]
    title = passage['title']
    retrieved_titles.append(title)
    is_supporting = title in supporting_titles
    pass
    pass
    pass
    pass
hits = set(retrieved_titles) & supporting_titles
pass
pass
pass
