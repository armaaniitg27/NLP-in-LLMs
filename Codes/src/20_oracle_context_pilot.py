import json
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
ROOT = Path(__file__).resolve().parent.parent
VARIANTS_FILE = ROOT / 'data' / 'prompt_variants.jsonl'
METADATA_FILE = ROOT / 'data' / 'passage_metadata.jsonl'
PREVIOUS_PILOT_FILE = ROOT / 'data' / 'answer_generation_pilot_v2.csv'
OUTPUT_FILE = ROOT / 'data' / 'oracle_context_pilot.csv'
MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'
MAX_NEW_TOKENS = 120
MAX_PASSAGE_CHARS = 2500
VARIANT_NAMES = ['original', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3']

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]

def get_passage_text(passage):
    for key in ['text', 'passage', 'content']:
        value = passage.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ''

def main():
    pass
    previous_df = pd.read_csv(PREVIOUS_PILOT_FILE)
    first_five_ids = previous_df['question_id'].drop_duplicates().head(5).tolist()
    question_records = load_jsonl(VARIANTS_FILE)
    passages = load_jsonl(METADATA_FILE)
    questions_by_id = {str(record.get('id', record.get('_id', ''))): record for record in question_records}
    passages_by_title = {}
    for passage in passages:
        title = passage.get('title')
        text = get_passage_text(passage)
        if title and text:
            passages_by_title[str(title)] = text
    reference_rows = previous_df.drop_duplicates('question_id').set_index('question_id')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    results = []
    for question_number, question_id in enumerate(first_five_ids, start=1):
        question_data = questions_by_id.get(str(question_id))
        if question_data is None:
            pass
            continue
        reference_row = reference_rows.loc[question_id]
        supporting_titles = json.loads(reference_row['supporting_titles'])
        question = question_data.get('question', reference_row['question'])
        variants = question_data.get('variants', {})
        context_documents = []
        missing_titles = []
        for title in supporting_titles:
            text = passages_by_title.get(str(title))
            if not text:
                missing_titles.append(title)
                continue
            context_documents.append(f'Document: {title}\n{text[:MAX_PASSAGE_CHARS]}')
        context = '\n\n'.join(context_documents)
        pass
        pass
        if missing_titles:
            pass
        for variant_name in VARIANT_NAMES:
            if variant_name == 'original':
                query = question
            else:
                query = variants.get(variant_name, question)
            system_prompt = "You are answering a question using supplied reference documents. Use only information supported by those documents. Do not invent facts or citations. If the documents do not establish the answer, reply exactly: 'Insufficient evidence in supplied documents.' Otherwise, answer concisely in one sentence and identify the document or documents supporting the answer. If the evidence is ambiguous or does not justify the comparison being asked, state that clearly."
            user_prompt = f'Question: {query}\n\nReference documents:\n{context}\n\nAnswer the question using only these documents.'
            messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=4096)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id)
            generated_tokens = output[:, inputs['input_ids'].shape[1]:]
            answer = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
            results.append({'question_id': question_id, 'variant': variant_name, 'question': question, 'query_used': query, 'reference_answer': reference_row['reference_answer'], 'supporting_titles': json.dumps(supporting_titles), 'supplied_titles': json.dumps([doc.split('\n', 1)[0].replace('Document: ', '') for doc in context_documents]), 'missing_titles': json.dumps(missing_titles), 'supplied_context': context, 'generated_answer': answer})
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
            pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
