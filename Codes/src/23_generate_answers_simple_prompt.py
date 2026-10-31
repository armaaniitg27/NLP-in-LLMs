import json
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / 'data' / 'curated_retrieval_results.csv'
METADATA_FILE = ROOT / 'data' / 'passage_metadata.jsonl'
PREVIOUS_PILOT_FILE = ROOT / 'data' / 'answer_generation_pilot_v2.csv'
OUTPUT_FILE = ROOT / 'data' / 'answer_generation_simple_prompt.csv'
MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'
TOP_K = 5
MAX_PASSAGE_CHARS = 1500
MAX_NEW_TOKENS = 120
VARIANTS = ['original', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3']

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]

def get_text(record):
    for key in ['text', 'passage', 'content']:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ''

def parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []

def main():
    pass
    retrieval_df = pd.read_csv(RESULTS_FILE)
    previous_df = pd.read_csv(PREVIOUS_PILOT_FILE)
    question_ids = previous_df['question_id'].drop_duplicates().head(5).tolist()
    retrieval_df['question_id'] = retrieval_df['question_id'].astype(str)
    retrieval_df['variant'] = retrieval_df['variant'].astype(str)
    question_ids = [str(qid) for qid in question_ids]
    passages = load_jsonl(METADATA_FILE)
    passages_by_title = {}
    for passage in passages:
        title = passage.get('title')
        text = get_text(passage)
        if title and text:
            passages_by_title[str(title)] = text
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    results = []
    for question_number, question_id in enumerate(question_ids, start=1):
        question_rows = retrieval_df[retrieval_df['question_id'] == question_id]
        if question_rows.empty:
            pass
            continue
        pass
        for variant in VARIANTS:
            matches = question_rows[question_rows['variant'] == variant]
            if matches.empty:
                pass
                continue
            row = matches.iloc[0]
            query = row['query']
            retrieved_titles = parse_list(row['retrieved_titles'])
            context_documents = []
            missing_titles = []
            truncated_titles = []
            for title in retrieved_titles[:TOP_K]:
                text = passages_by_title.get(str(title))
                if not text:
                    missing_titles.append(title)
                    continue
                if len(text) > MAX_PASSAGE_CHARS:
                    truncated_titles.append(title)
                context_documents.append(f'Document: {title}\n{text[:MAX_PASSAGE_CHARS]}')
            context = '\n\n'.join(context_documents)
            messages = [{'role': 'system', 'content': "Answer the user's question using the information in the documents. Give a short, direct answer."}, {'role': 'user', 'content': f'Question: {query}\n\nDocuments:\n{context}\n\nAnswer:'}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=4096)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id)
            new_tokens = output[:, inputs['input_ids'].shape[1]:]
            answer = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
            results.append({'question_id': question_id, 'variant': variant, 'question': row.get('question', query), 'query_used': query, 'reference_answer': row.get('reference_answer', ''), 'supporting_titles': row.get('supporting_titles', ''), 'retrieved_titles': json.dumps(retrieved_titles), 'missing_titles': json.dumps(missing_titles), 'truncated_titles': json.dumps(truncated_titles), 'retrieved_context': context, 'generated_answer': answer, 'manual_correctness_assessment': '', 'manual_evidence_support_assessment': '', 'manual_abstention_assessment': '', 'manual_failure_category': '', 'manual_audit_notes': ''})
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
            pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
