import json
import csv
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / 'data' / 'selected_30_questions.jsonl'
OUTPUT_FILE = ROOT / 'data' / 'generated_paraphrase_candidates.csv'
MODELS = ['humarin/chatgpt_paraphraser_on_T5_base', 'prithivida/parrot_paraphraser_on_T5']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
pass
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    questions = [json.loads(line) for line in f if line.strip()]
results = []
for model_name in MODELS:
    pass
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    model.eval()
    for i, item in enumerate(questions, start=1):
        question = item['question']
        inputs = tokenizer(f'paraphrase: {question}', return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, do_sample=True, top_p=0.9, temperature=0.9, num_return_sequences=3, repetition_penalty=1.5, no_repeat_ngram_size=2, max_new_tokens=96)
        candidates = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        seen = set()
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate or candidate.lower() in seen:
                continue
            seen.add(candidate.lower())
            results.append({'question_id': item.get('id', ''), 'original_question': question, 'model': model_name, 'candidate': candidate, 'meaning_preserved': '', 'meaningful_rewording': '', 'grammatical': '', 'accepted': '', 'reviewer_notes': ''})
        pass
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
fields = ['question_id', 'original_question', 'model', 'candidate', 'meaning_preserved', 'meaningful_rewording', 'grammatical', 'accepted', 'reviewer_notes']
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)
pass
pass
