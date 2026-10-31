import json
import re
import csv
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
INPUT_FILE = DATA_DIR / 'prompt_variants.jsonl'
OUTPUT_FILE = DATA_DIR / 'paraphrase_candidates.jsonl'
REVIEW_FILE = DATA_DIR / 'paraphrase_review.csv'
MODEL_NAME = 'Vamsi/T5_Paraphrase_Paws'
NUM_PARAPHRASES = 3
BATCH_SIZE = 4
RANDOM_SEED = 42

def load_jsonl(path):
    with path.open('r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def extract_capitalized_tokens(text):
    tokens = re.findall("\\b[A-Z][A-Za-z'-]*\\b", text)
    return set((token.lower() for token in tokens[1:]))

def validate_candidate(original, candidate):
    original_entities = extract_capitalized_tokens(original)
    candidate_tokens = set(re.findall("\\b[A-Za-z'-]+\\b", candidate.lower()))
    missing_tokens = sorted((token for token in original_entities if token not in candidate_tokens))
    original_numbers = set(re.findall('\\d+', original))
    candidate_numbers = set(re.findall('\\d+', candidate))
    missing_numbers = sorted(original_numbers - candidate_numbers)
    return {'missing_capitalized_tokens': missing_tokens, 'missing_numbers': missing_numbers, 'review_needed': bool(missing_tokens or missing_numbers)}

def main():
    torch.manual_seed(RANDOM_SEED)
    examples = load_jsonl(INPUT_FILE)
    pass
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    records = []
    for start in range(0, len(examples), BATCH_SIZE):
        batch = examples[start:start + BATCH_SIZE]
        inputs = ['paraphrase: ' + example['question'] + ' </s>' for example in batch]
        encoded = tokenizer(inputs, padding=True, truncation=True, max_length=256, return_tensors='pt')
        with torch.no_grad():
            generated = model.generate(**encoded, max_length=128, num_beams=10, num_return_sequences=NUM_PARAPHRASES, do_sample=False, early_stopping=True)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for i, example in enumerate(batch):
            candidates = decoded[i * NUM_PARAPHRASES:(i + 1) * NUM_PARAPHRASES]
            cleaned = []
            seen = {example['question'].strip().lower()}
            for candidate in candidates:
                candidate = ' '.join(candidate.strip().split())
                if not candidate:
                    continue
                key = candidate.lower()
                if key in seen:
                    continue
                seen.add(key)
                validation = validate_candidate(example['question'], candidate)
                cleaned.append({'text': candidate, **validation})
            records.append({'id': example['id'], 'question': example['question'], 'answer': example['answer'], 'supporting_titles': example['supporting_titles'], 'paraphrases': cleaned})
        pass
    with OUTPUT_FILE.open('w', encoding='utf-8') as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + '\n')
    review_rows = []
    for record in records:
        row = {'id': record['id'], 'original_question': record['question'], 'answer': record['answer'], 'supporting_titles': ' | '.join(record['supporting_titles'])}
        for i in range(NUM_PARAPHRASES):
            if i < len(record['paraphrases']):
                item = record['paraphrases'][i]
                row[f'paraphrase_{i + 1}'] = item['text']
                row[f'missing_tokens_{i + 1}'] = ', '.join(item['missing_capitalized_tokens'])
                row[f'review_needed_{i + 1}'] = item['review_needed']
            else:
                row[f'paraphrase_{i + 1}'] = ''
                row[f'missing_tokens_{i + 1}'] = ''
                row[f'review_needed_{i + 1}'] = True
        review_rows.append(row)
    fieldnames = list(review_rows[0].keys()) if review_rows else []
    with REVIEW_FILE.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)
    total_candidates = sum((len(record['paraphrases']) for record in records))
    flagged = sum((item['review_needed'] for record in records for item in record['paraphrases']))
    pass
    pass
    pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
