import json
import csv
import random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / 'data' / 'prompt_variants.jsonl'
SELECTED_FILE = ROOT / 'data' / 'selected_30_questions.jsonl'
REVIEW_FILE = ROOT / 'data' / 'paraphrase_review_template.csv'
SEED = 42
N_QUESTIONS = 30
N_PARAPHRASES = 3
if not INPUT_FILE.exists():
    raise FileNotFoundError()
with INPUT_FILE.open('r', encoding='utf-8') as f:
    examples = [json.loads(line) for line in f if line.strip()]
if len(examples) < N_QUESTIONS:
    raise ValueError()
rng = random.Random(SEED)
selected_indices = sorted(rng.sample(range(len(examples)), N_QUESTIONS))
selected = [examples[i] for i in selected_indices]
with SELECTED_FILE.open('w', encoding='utf-8') as f:
    for example in selected:
        f.write(json.dumps(example, ensure_ascii=False) + '\n')
fieldnames = ['question_id', 'original_question', 'candidate_number', 'candidate_paraphrase', 'generated_by', 'meaning_preserved', 'meaningful_rewording', 'grammatical', 'accepted', 'reviewer_notes']
with REVIEW_FILE.open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i, example in enumerate(selected, start=1):
        question_id = example.get('id', f'sample_{i:03d}')
        question = example.get('question', '')
        for candidate_number in range(1, N_PARAPHRASES + 1):
            writer.writerow({'question_id': question_id, 'original_question': question, 'candidate_number': candidate_number, 'candidate_paraphrase': '', 'generated_by': '', 'meaning_preserved': '', 'meaningful_rewording': '', 'grammatical': '', 'accepted': '', 'reviewer_notes': ''})
pass
pass
pass
pass
pass
