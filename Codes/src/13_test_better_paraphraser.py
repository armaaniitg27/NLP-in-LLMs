import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / 'data' / 'prompt_variants.jsonl'
MODEL_NAME = 'humarin/chatgpt_paraphraser_on_T5_base'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    prompt = f'paraphrase: {question}'
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, num_beams=5, num_beam_groups=5, num_return_sequences=3, diversity_penalty=3.0, repetition_penalty=2.0, no_repeat_ngram_size=2, max_length=128, early_stopping=True)
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
output_file = ROOT / 'data' / 'humarin_paraphrase_test.txt'
with open(output_file, 'w', encoding='utf-8') as file:
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
