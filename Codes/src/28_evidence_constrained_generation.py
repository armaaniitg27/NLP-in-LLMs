import ast
import json
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RETRIEVAL_FILE = os.path.join(RESULTS_DIR, "retrieval_results_28q_rerun.csv")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "evidence_constrained_results_28q.csv")
MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
MAX_NEW_TOKENS = 48
PASSAGE_CHAR_LIMIT = 1500
MAX_INPUT_LENGTH = 8192
EXPECTED_ROWS = 112

def parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value) or not str(value).strip():
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        parsed = ast.literal_eval(value)
    return parsed if isinstance(parsed, list) else []

def main():
    if not torch.cuda.is_available():
        pass
    pass
    pass
    if not os.path.exists(RETRIEVAL_FILE):
        pass
    retrieval = pd.read_csv(RETRIEVAL_FILE)
    required = {'question_id', 'variant', 'original_question', 'query_used', 'reference_answer', 'supporting_titles', 'retrieved_titles', 'retrieved_passages'}
    missing = required - set(retrieval.columns)
    if missing:
        pass
    if len(retrieval) != EXPECTED_ROWS:
        pass
    if retrieval.duplicated(['question_id', 'variant']).any():
        pass
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
            expected = set(zip(retrieval['question_id'].astype(str), retrieval['variant']))
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
    system_prompt = 'Answer the question using only the supplied documents. Every factual claim in your answer must be supported by those documents. Do not use outside knowledge or make assumptions. Ignore document details that do not help answer the question. If the documents do not provide enough evidence for a reliable answer, explicitly say that the information is insufficient. Do not guess. Give a short, direct answer.'
    for _, row in retrieval.iterrows():
        question_id = str(row['question_id'])
        variant = row['variant']
        key = (question_id, variant)
        if key in completed:
            continue
        query = str(row['query_used'])
        passages = parse_list(row['retrieved_passages'])
        context_blocks = []
        truncated_titles = []
        for rank, passage in enumerate(passages, start=1):
            title = str(passage.get('title', f'Document {rank}'))
            text = passage.get('text', '')
            if isinstance(text, list):
                text = ' '.join(map(str, text))
            else:
                text = str(text)
            if len(text) > PASSAGE_CHAR_LIMIT:
                text = text[:PASSAGE_CHAR_LIMIT]
                truncated_titles.append(title)
            context_blocks.append(f'[Document {rank}: {title}]\n{text}')
        if not context_blocks:
            pass
        context = '\n\n'.join(context_blocks)
        user_prompt = f'Question: {query}\n\nDocuments:\n{context}\n\nAnswer using only the documents:'
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=MAX_INPUT_LENGTH).to('cuda')
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id)
        new_tokens = output[0][inputs['input_ids'].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append({'model_name': MODEL_NAME, 'method': 'evidence_constrained_generation', 'question_id': question_id, 'variant': variant, 'original_question': str(row['original_question']), 'query_used': query, 'reference_answer': row['reference_answer'], 'supporting_titles': json.dumps(parse_list(row['supporting_titles']), ensure_ascii=False), 'retrieved_titles': json.dumps(parse_list(row['retrieved_titles']), ensure_ascii=False), 'retrieved_context': context, 'truncated_titles': json.dumps(truncated_titles, ensure_ascii=False), 'generated_answer': answer})
        completed.add(key)
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
