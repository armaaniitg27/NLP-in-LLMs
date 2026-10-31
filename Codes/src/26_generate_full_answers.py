import ast
import json
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
DATA_DIR = 'data'
RETRIEVAL_FILE = os.path.join(DATA_DIR, 'curated_retrieval_results.csv')
QUESTIONS_FILE = os.path.join(DATA_DIR, 'prompt_variants.jsonl')
PASSAGES_FILE = os.path.join(DATA_DIR, 'passage_metadata.jsonl')
OUTPUT_FILE = os.path.join(DATA_DIR, 'full_answer_generation_results_0.5B.csv')
MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
MAX_NEW_TOKENS = 48
PASSAGE_CHAR_LIMIT = 1500
MAX_INPUT_LENGTH = 8192

def parse_list(value):
    """Convert a CSV list representation into a Python list."""
    if isinstance(value, list):
        return value
    if pd.isna(value) or not str(value).strip():
        return []
    try:
        result = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        result = ast.literal_eval(value)
    return result if isinstance(result, list) else []

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]

def main():
    if not torch.cuda.is_available():
        raise RuntimeError()
    device = torch.device('cuda')
    pass
    pass
    retrieval = pd.read_csv(RETRIEVAL_FILE)
    questions = load_jsonl(QUESTIONS_FILE)
    passages = load_jsonl(PASSAGES_FILE)
    question_lookup = {str(item['id']): item for item in questions}
    passage_lookup = {item['title']: item for item in passages}
    missing_ids = sorted(set(retrieval['question_id'].astype(str)) - set(question_lookup))
    if missing_ids:
        raise ValueError()
    if len(retrieval) != 120:
        raise ValueError()
    if retrieval.duplicated(['question_id', 'variant']).any():
        raise ValueError()
    pass
    pass
    pass
    pass
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    model.eval()
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        if not existing.empty and 'model_name' not in existing.columns:
            raise ValueError()
        if not existing.empty and (not existing['model_name'].eq(MODEL_NAME).all()):
            raise ValueError()
        completed = {(str(row.question_id), row.variant) for row in existing.itertuples()}
        results = existing.to_dict('records')
        pass
    else:
        completed = set()
        results = []
    for _, row in retrieval.iterrows():
        question_id = str(row['question_id'])
        variant = row['variant']
        run_key = (question_id, variant)
        if run_key in completed:
            continue
        source = question_lookup[question_id]
        reference_answer = source['answer']
        original_question = source['question']
        query = row['query']
        retrieved_titles = parse_list(row['retrieved_titles'])
        supporting_titles = source.get('supporting_titles', [])
        context_blocks = []
        missing_titles = []
        truncated_titles = []
        for rank, title in enumerate(retrieved_titles, start=1):
            passage = passage_lookup.get(title)
            if passage is None:
                missing_titles.append(title)
                continue
            text = passage.get('text', '')
            if not text:
                text = passage.get('sentences', '')
            if isinstance(text, list):
                text = ' '.join(map(str, text))
            else:
                text = str(text)
            if len(text) > PASSAGE_CHAR_LIMIT:
                text = text[:PASSAGE_CHAR_LIMIT]
                truncated_titles.append(title)
            context_blocks.append(f'[Document {rank}: {title}]\n{text}')
        context = '\n\n'.join(context_blocks)
        system_prompt = "Answer the user's question using the information in the documents. Give a short, direct answer. Do not deliberately guess when the documents do not provide enough information."
        user_prompt = f'Question: {query}\n\nDocuments:\n{context}\n\nAnswer:'
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=MAX_INPUT_LENGTH).to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id)
        new_tokens = output[0][inputs['input_ids'].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        result = {'model_name': MODEL_NAME, 'question_id': question_id, 'variant': variant, 'original_question': original_question, 'query_used': query, 'reference_answer': reference_answer, 'supporting_titles': json.dumps(supporting_titles), 'retrieved_titles': json.dumps(retrieved_titles), 'missing_titles': json.dumps(missing_titles), 'truncated_titles': json.dumps(truncated_titles), 'retrieved_context': context, 'generated_answer': answer, 'manual_correctness_assessment': '', 'manual_evidence_support_assessment': '', 'manual_abstention_assessment': '', 'manual_failure_category': '', 'manual_audit_notes': ''}
        results.append(result)
        completed.add(run_key)
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
