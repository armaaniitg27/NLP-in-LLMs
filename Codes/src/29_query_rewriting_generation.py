import ast
import json
import os
import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RETRIEVAL_FILE = os.path.join(RESULTS_DIR, "retrieval_results_28q_rerun.csv")
METADATA_FILE = os.path.join(DATA_DIR, "passage_metadata.jsonl")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "query_rewriting_results_28q.csv")
MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EXPECTED_ROWS = 112
TOP_K = 5
PASSAGE_CHAR_LIMIT = 1500
MAX_INPUT_LENGTH = 8192

def parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value) or not str(value).strip():
        return []
    try:
        result = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        result = ast.literal_eval(value)
    return result if isinstance(result, list) else []

def generate_text(model, tokenizer, system_prompt, user_prompt, max_tokens):
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=MAX_INPUT_LENGTH).to('cuda')
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id)
    new_tokens = output[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def load_passages():
    if not os.path.exists(METADATA_FILE):
        pass
    passages = []
    with open(METADATA_FILE, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                item = json.loads(line)
                title = str(item.get('title', ''))
                passage_text = item.get('text', '')
                if isinstance(passage_text, list):
                    passage_text = ' '.join(map(str, passage_text))
                else:
                    passage_text = str(passage_text)
                if not passage_text.strip():
                    passage_text = str(item.get('passage_text', ''))
                if not passage_text.strip():
                    passage_text = str(item.get('content', ''))
                passages.append({'title': title, 'text': passage_text})
    if not passages:
        pass
    if not any((p['text'].strip() for p in passages)):
        pass
    pass
    pass
    return passages

def build_index(passages, embedder):
    documents = [(p['title'] + ' ' + p['text']).strip() for p in passages]
    pass
    embeddings = embedder.encode(documents, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(query, embedder, index, passages):
    query_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    scores, indices = index.search(query_embedding, TOP_K)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        passage = passages[int(idx)]
        results.append({'title': passage['title'], 'text': passage['text'], 'score': float(score)})
    return results

def main():
    if not torch.cuda.is_available():
        pass
    pass
    pass
    pass
    retrieval_df = pd.read_csv(RETRIEVAL_FILE)
    required = {'question_id', 'variant', 'original_question', 'query_used', 'reference_answer', 'supporting_titles'}
    missing = required - set(retrieval_df.columns)
    if missing:
        pass
    if len(retrieval_df) != EXPECTED_ROWS:
        pass
    if retrieval_df.duplicated(['question_id', 'variant']).any():
        pass
    if retrieval_df[['question_id', 'variant', 'query_used']].isna().any().any():
        pass
    pass
    passages = load_passages()
    pass
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    index = build_index(passages, embedder)
    pass
    pass
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, low_cpu_mem_usage=True).to('cuda')
    model.eval()
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        if not existing.empty:
            if 'model_name' not in existing.columns:
                pass
            if not existing['model_name'].eq(MODEL_NAME).all():
                pass
            if existing.duplicated(['question_id', 'variant']).any():
                pass
            expected = set(zip(retrieval_df['question_id'].astype(str), retrieval_df['variant']))
            completed = set(zip(existing['question_id'].astype(str), existing['variant']))
            if not completed.issubset(expected):
                pass
            results = existing.to_dict('records')
            pass
        else:
            results = []
            completed = set()
    else:
        results = []
        completed = set()
    rewrite_system = "You are a search-query rewriting assistant. Rewrite the user's question into one concise query suitable for retrieving relevant passages from a document collection. Preserve the original question's intent, named entities, relationships, dates, and requested facts. Do not answer the question. Do not introduce new entities or assumptions. Output only the rewritten search query."
    answer_system = 'Answer the original question using the supplied documents. Give a short, direct answer. Do not deliberately guess when the documents do not provide enough information.'
    for _, row in retrieval_df.iterrows():
        question_id = str(row['question_id'])
        variant = row['variant']
        key = (question_id, variant)
        if key in completed:
            continue
        original_question = str(row['original_question'])
        original_query = str(row['query_used'])
        rewritten_query = generate_text(model, tokenizer, rewrite_system, f'Original question:\n{original_query}\n\nRewritten search query:', max_tokens=64)
        rewritten_query = rewritten_query.strip().strip('"').strip()
        if not rewritten_query:
            rewritten_query = original_query
        retrieved = retrieve(rewritten_query, embedder, index, passages)
        if not retrieved:
            pass
        context_blocks = []
        truncated_titles = []
        for rank, passage in enumerate(retrieved, start=1):
            title = passage['title']
            text = passage['text']
            if len(text) > PASSAGE_CHAR_LIMIT:
                text = text[:PASSAGE_CHAR_LIMIT]
                truncated_titles.append(title)
            context_blocks.append(f'[Document {rank}: {title}]\n{text}')
        context = '\n\n'.join(context_blocks)
        answer_prompt = f"Original question: {original_question}\n\nDocuments retrieved using the rewritten query '{rewritten_query}':\n\n{context}\n\nAnswer the original question:"
        answer = generate_text(model, tokenizer, answer_system, answer_prompt, max_tokens=48)
        results.append({'model_name': MODEL_NAME, 'method': 'query_rewriting', 'question_id': question_id, 'variant': variant, 'original_question': original_question, 'original_query': original_query, 'rewritten_query': rewritten_query, 'reference_answer': row['reference_answer'], 'supporting_titles': json.dumps(parse_list(row['supporting_titles']), ensure_ascii=False), 'retrieved_titles': json.dumps([p['title'] for p in retrieved], ensure_ascii=False), 'retrieved_scores': json.dumps([p['score'] for p in retrieved]), 'retrieved_context': context, 'truncated_titles': json.dumps(truncated_titles, ensure_ascii=False), 'generated_answer': answer})
        completed.add(key)
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
