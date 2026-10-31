import json
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RETRIEVAL_FILE = DATA / 'curated_retrieval_results.csv'
METADATA_FILE = DATA / 'passage_metadata.jsonl'
QUESTIONS_FILE = DATA / 'prompt_variants.jsonl'
OUTPUT_FILE = DATA / 'answer_generation_pilot.csv'
MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'
TOP_K = 5
MAX_CHARS_PER_PASSAGE = 1500
MAX_NEW_TOKENS = 100
VARIANT_ORDER = {'original': 0, 'paraphrase_1': 1, 'paraphrase_2': 2, 'paraphrase_3': 3}

def load_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records

def main():
    retrieval = pd.read_csv(RETRIEVAL_FILE)
    passage_records = load_jsonl(METADATA_FILE)
    question_records = load_jsonl(QUESTIONS_FILE)
    passage_lookup = {item['title']: item['text'] for item in passage_records}
    question_lookup = {item['id']: item for item in question_records}
    original_rows = retrieval[retrieval['variant'] == 'original']
    selected_ids = original_rows['question_id'].drop_duplicates().head(5).tolist()
    pilot = retrieval[retrieval['question_id'].isin(selected_ids)].copy()
    pilot['variant_order'] = pilot['variant'].map(VARIANT_ORDER)
    pilot = pilot.sort_values(['question_id', 'variant_order'])
    if len(pilot) != 20:
        raise ValueError()
    pass
    pass
    pass
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.to('cpu')
    model.eval()
    results = []
    for run_number, (_, row) in enumerate(pilot.iterrows(), start=1):
        question_id = row['question_id']
        variant = row['variant']
        query = row['query']
        retrieved_titles = json.loads(row['retrieved_titles'])
        context_blocks = []
        missing_titles = []
        truncated_titles = []
        for rank, title in enumerate(retrieved_titles[:TOP_K], start=1):
            text = passage_lookup.get(title)
            if text is None:
                missing_titles.append(title)
                continue
            if len(text) > MAX_CHARS_PER_PASSAGE:
                truncated_titles.append(title)
            excerpt = text[:MAX_CHARS_PER_PASSAGE]
            context_blocks.append(f'[Document {rank}: {title}]\n{excerpt}')
        context = '\n\n'.join(context_blocks)
        messages = [{'role': 'system', 'content': 'Answer the question using only the supplied documents. Do not add facts from outside the documents. If the documents do not provide enough evidence, explicitly say that the evidence is insufficient. Give a concise answer.'}, {'role': 'user', 'content': f'Question: {query}\n\nRetrieved documents:\n{context}\n\nAnswer:'}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=False)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        new_tokens = output_ids[0, inputs['input_ids'].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        reference = question_lookup.get(question_id, {})
        result = {'question_id': question_id, 'variant': variant, 'query': query, 'reference_answer': reference.get('answer', ''), 'supporting_titles': row['supporting_titles'], 'retrieved_titles': json.dumps(retrieved_titles[:TOP_K], ensure_ascii=False), 'retrieval_recall_at_5': row['recall_at_5'], 'complete_support_retrieval': row['complete_support_retrieval'], 'context': context, 'context_truncated_titles': json.dumps(truncated_titles, ensure_ascii=False), 'missing_passage_titles': json.dumps(missing_titles, ensure_ascii=False), 'generated_answer': answer, 'answer_correctness': '', 'evidence_support': '', 'failure_category': ''}
        results.append(result)
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        pass
        pass
    pass
    pass
if __name__ == '__main__':
    main()
