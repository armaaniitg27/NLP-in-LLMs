import json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
index = faiss.read_index(str(DATA_DIR / 'passages.index'))
with (DATA_DIR / 'passage_metadata.jsonl').open('r', encoding='utf-8') as f:
    passages = [json.loads(line) for line in f]
with (DATA_DIR / 'hotpotqa_sample.jsonl').open('r', encoding='utf-8') as f:
    example = json.loads(next(f))
supporting_titles = set(example['supporting_facts']['title'])
variants = {'original': example['question'], 'paraphrase': "Between Arthur's Magazine and First for Women, which was founded earlier?", 'conversational': "I want to know which of these two magazines came first: Arthur's Magazine or First for Women.", 'reordered': "Was First for Women or Arthur's Magazine established first?", 'expanded': "Compare the founding dates of Arthur's Magazine and First for Women. Which one began publication earlier?"}
pass
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
results = {}
for name, question in variants.items():
    embedding = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(embedding, 5)
    retrieved_titles = [passages[int(i)]['title'] for i in indices[0]]
    hits = set(retrieved_titles) & supporting_titles
    recall = len(hits) / len(supporting_titles)
    results[name] = {'question': question, 'titles': retrieved_titles, 'scores': scores[0].tolist(), 'recall': recall}
original_titles = set(results['original']['titles'])
pass
for name, result in results.items():
    overlap = len(original_titles & set(result['titles'])) / 5
    pass
    pass
    pass
    pass
    for rank, (title, score) in enumerate(zip(result['titles'], result['scores']), start=1):
        pass
output_path = DATA_DIR / 'prompt_variability_results.json'
with output_path.open('w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
pass
