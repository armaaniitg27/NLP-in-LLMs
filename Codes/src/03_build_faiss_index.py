import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
input_path = DATA_DIR / 'passages.jsonl'
index_path = DATA_DIR / 'passages.index'
metadata_path = DATA_DIR / 'passage_metadata.jsonl'
pass
passages = []
with input_path.open('r', encoding='utf-8') as f:
    for line in f:
        passages.append(json.loads(line))
pass
pass
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
texts = [p['title'] + '. ' + p['text'] for p in passages]
pass
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
embeddings = np.asarray(embeddings, dtype=np.float32)
pass
pass
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, str(index_path))
with metadata_path.open('w', encoding='utf-8') as f:
    for passage in passages:
        f.write(json.dumps(passage, ensure_ascii=False) + '\n')
pass
pass
pass
pass
