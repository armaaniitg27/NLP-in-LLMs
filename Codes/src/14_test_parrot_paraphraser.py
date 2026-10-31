import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / 'data' / 'prompt_variants.jsonl'
OUTPUT_FILE = ROOT / 'data' / 'parrot_paraphrase_test.txt'
MODEL_NAME = 'prithivida/parrot_paraphraser_on_T5'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
pass
pass
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
pass
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

def generate_paraphrases(question):
    prompt = f'paraphrase: {question}'
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, do_sample=True, top_p=0.9, temperature=0.9, num_return_sequences=3, repetition_penalty=1.5, no_repeat_ngram_size=2, max_new_tokens=96)
    candidates = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    cleaned = []
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in cleaned:
            cleaned.append(candidate)
    return cleaned
with open(INPUT_FILE, 'r', encoding='utf-8') as file:
    examples = [json.loads(line) for line in file if line.strip()]
examples = examples[:10]
with open(OUTPUT_FILE, 'w', encoding='utf-8') as file:
    for i, example in enumerate(examples, start=1):
        question = example['question']
        candidates = generate_paraphrases(question)
        file.write(f'\nQUESTION {i}\n')
        file.write(f'Original: {question}\n')
        for j, candidate in enumerate(candidates, start=1):
            file.write(f'Candidate {j}: {candidate}\n')
        file.write('-' * 70 + '\n')
        pass
pass
