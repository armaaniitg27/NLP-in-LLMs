import json
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
RESULTS_FILE = DATA / 'curated_retrieval_results.csv'
METADATA_FILE = DATA / 'passage_metadata.jsonl'
QUESTIONS_FILE = DATA / 'prompt_variants.jsonl'
OUTPUT_FILE = DATA / 'answer_generation_pilot_v2.csv'
MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'
TOP_K = 5
MAX_PASSAGE_CHARS = 1500
MAX_NEW_TOKENS = 160

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]

def get_value(row, *names, default=None):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default

def main():
    results = pd.read_csv(RESULTS_FILE)
    metadata_rows = load_jsonl(METADATA_FILE)
    question_rows = load_jsonl(QUESTIONS_FILE)
    metadata = {}
    for row in metadata_rows:
        title = row.get('title')
        if title is not None:
            metadata[title] = row
    questions = {row['id']: row for row in question_rows}
    original_rows = results[results['variant'] == 'original']
    selected_ids = original_rows['question_id'].drop_duplicates().head(5).tolist()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    model.to('cpu')
    output_rows = []
    for question_id in selected_ids:
        question_data = questions.get(question_id)
        if question_data is None:
            pass
            continue
        question_result_rows = results[results['question_id'] == question_id]
        variants = question_data.get('variants', {})
        for _, result in question_result_rows.iterrows():
            variant = result['variant']
            query = get_value(result, 'query', 'prompt', 'retrieval_query', default=None)
            if query is None:
                query = variants.get(variant, question_data['question'])
            titles_value = get_value(result, 'retrieved_titles', 'top_titles', 'retrieved_passages', default=[])
            if isinstance(titles_value, str):
                try:
                    titles_value = json.loads(titles_value)
                except json.JSONDecodeError:
                    titles_value = [title.strip() for title in titles_value.split('|') if title.strip()]
            if not isinstance(titles_value, list):
                titles_value = []
            context_parts = []
            missing_titles = []
            truncated_titles = []
            for rank, title in enumerate(titles_value[:TOP_K], start=1):
                passage = metadata.get(title)
                if passage is None:
                    missing_titles.append(title)
                    continue
                text = passage.get('text', passage.get('passage', passage.get('content', '')))
                if not isinstance(text, str) or not text:
                    missing_titles.append(title)
                    continue
                if len(text) > MAX_PASSAGE_CHARS:
                    text = text[:MAX_PASSAGE_CHARS]
                    truncated_titles.append(title)
                context_parts.append(f'[Document {rank}: {title}]\n{text}')
            context = '\n\n'.join(context_parts)
            system_prompt = "You answer questions using retrieved documents. Use only facts directly supported by those documents. Do not use outside knowledge, guess, or fill in missing facts. If the documents do not establish the answer, respond exactly: 'Insufficient evidence in retrieved documents.' If you can answer, give at most two sentences and include one short exact quotation from the documents that supports your answer. Never invent or alter a quotation."
            user_prompt = f'Question: {query}\n\nRetrieved documents:\n{context}\n\nAnswer according to the instructions. If the evidence is insufficient, use the required abstention sentence.'
            messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=4096)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id)
            new_tokens = generated[0][inputs['input_ids'].shape[1]:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            row = {'question_id': question_id, 'variant': variant, 'question': question_data['question'], 'query_used': query, 'reference_answer': question_data.get('answer', ''), 'supporting_titles': json.dumps(question_data.get('supporting_titles', [])), 'retrieved_titles': json.dumps(titles_value[:TOP_K]), 'missing_titles': json.dumps(missing_titles), 'truncated_titles': json.dumps(truncated_titles), 'retrieved_context': context, 'generated_answer': answer, 'manual_correctness_assessment': '', 'manual_evidence_support_assessment': '', 'manual_abstention_assessment': '', 'manual_failure_category': '', 'manual_audit_notes': ''}
            output_rows.append(row)
            pd.DataFrame(output_rows).to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
            pass
            pass
    pass
    pass
if __name__ == '__main__':
    main()
