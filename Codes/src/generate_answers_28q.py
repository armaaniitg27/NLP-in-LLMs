import ast
import json
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
DATA_DIR = 'data'
RETRIEVAL_FILE = os.path.join(DATA_DIR, 'reranking_results_28q.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'reranked_answer_generation_results_28q.csv')
MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
MAX_NEW_TOKENS = 48
PASSAGE_CHAR_LIMIT = 1500
MAX_INPUT_LENGTH = 8192
EXPECTED_QUESTIONS = 28
EXPECTED_ROWS = 112
EXPECTED_VARIANTS = {'original', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3'}

def parse_list(value):
    """Parse a JSON or Python-list representation from a CSV cell."""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    if not str(value).strip():
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    return parsed if isinstance(parsed, list) else []

def validate_dataset(retrieval):
    """Check that the reranking dataset has the expected structure."""
    required_columns = {'question_id', 'variant', 'original_question', 'query_used', 'reference_answer', 'supporting_titles', 'reranked_titles', 'reranked_passages'}
    missing_columns = required_columns - set(retrieval.columns)
    if missing_columns:
        raise ValueError()
    if len(retrieval) != EXPECTED_ROWS:
        raise ValueError()
    if retrieval['question_id'].nunique() != EXPECTED_QUESTIONS:
        raise ValueError()
    if set(retrieval['variant'].unique()) != EXPECTED_VARIANTS:
        raise ValueError()
    if retrieval.duplicated(['question_id', 'variant']).any():
        raise ValueError()
    essential_columns = ['question_id', 'variant', 'query_used', 'reranked_passages']
    if retrieval[essential_columns].isna().any().any():
        raise ValueError()
    for _, row in retrieval.iterrows():
        passages = parse_list(row['reranked_passages'])
        if not passages:
            raise ValueError()
        if len(passages) > 5:
            raise ValueError()
        if any((not isinstance(passage, dict) for passage in passages)):
            raise ValueError()
    pass
    pass
    pass

def main():
    if not os.path.exists(RETRIEVAL_FILE):
        raise FileNotFoundError()
    retrieval = pd.read_csv(RETRIEVAL_FILE)
    validate_dataset(retrieval)
    if not torch.cuda.is_available():
        raise RuntimeError()
    device = 'cuda'
    pass
    pass
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        if not existing.empty:
            if 'model_name' not in existing.columns:
                raise ValueError()
            if not existing['model_name'].eq(MODEL_NAME).all():
                raise ValueError()
            if existing.duplicated(['question_id', 'variant']).any():
                raise ValueError()
            expected_keys = set(zip(retrieval['question_id'].astype(str), retrieval['variant']))
            existing_keys = set(zip(existing['question_id'].astype(str), existing['variant']))
            if not existing_keys.issubset(expected_keys):
                raise ValueError()
            if 'retrieved_titles' not in existing.columns:
                raise ValueError()
            completed = existing_keys
            results = existing.to_dict('records')
            pass
        else:
            completed = set()
            results = []
    else:
        completed = set()
        results = []
    pass
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pass
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    model.eval()
    system_prompt = "Answer the user's question using the information in the documents. Give a short, direct answer. Do not deliberately guess when the documents do not provide enough information."
    for _, row in retrieval.iterrows():
        question_id = str(row['question_id'])
        variant = row['variant']
        run_key = (question_id, variant)
        if run_key in completed:
            continue
        query = str(row['query_used'])
        original_question = str(row['original_question'])
        supporting_titles = parse_list(row['supporting_titles'])
        reranked_titles = parse_list(row['reranked_titles'])
        reranked_passages = parse_list(row['reranked_passages'])
        context_blocks = []
        truncated_titles = []
        for rank, passage in enumerate(reranked_passages, start=1):
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
        context = '\n\n'.join(context_blocks)
        user_prompt = f'Question: {query}\n\nDocuments:\n{context}\n\nAnswer:'
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=MAX_INPUT_LENGTH).to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id)
        new_tokens = output[0:1, inputs['input_ids'].shape[1]:]
        answer = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        results.append({'model_name': MODEL_NAME, 'question_id': question_id, 'variant': variant, 'original_question': original_question, 'query_used': query, 'reference_answer': row['reference_answer'], 'supporting_titles': json.dumps(supporting_titles, ensure_ascii=False), 'retrieved_titles': json.dumps(reranked_titles, ensure_ascii=False), 'reranked_titles': json.dumps(reranked_titles, ensure_ascii=False), 'retrieved_context': context, 'truncated_titles': json.dumps(truncated_titles, ensure_ascii=False), 'generated_answer': answer, 'manual_correctness_assessment': '', 'manual_evidence_support_assessment': '', 'manual_abstention_assessment': '', 'manual_failure_category': '', 'manual_audit_notes': ''})
        completed.add(run_key)
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
