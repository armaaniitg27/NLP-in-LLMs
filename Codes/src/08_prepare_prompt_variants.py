import json
import random
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
INPUT_FILE = DATA_DIR / 'hotpotqa_sample.jsonl'
OUTPUT_FILE = DATA_DIR / 'prompt_variants.jsonl'
NUM_QUESTIONS = 100
RANDOM_SEED = 42

def get_supporting_titles(example):
    return list(dict.fromkeys(example['supporting_facts']['title']))

def create_variants(question):
    return {'original': question, 'polite': f'Could you answer this question: {question}', 'conversational': f'I would like to find out the answer to this: {question}', 'task_framed': f'Determine the answer to the following question: {question}', 'evidence_focused': f'Using relevant information, answer this question: {question}'}

def main():
    examples = []
    with INPUT_FILE.open('r', encoding='utf-8') as file:
        for line in file:
            examples.append(json.loads(line))
    rng = random.Random(RANDOM_SEED)
    selected = rng.sample(examples, NUM_QUESTIONS)
    records = []
    for example in selected:
        question = example['question']
        record = {'id': example['id'], 'question': question, 'answer': example['answer'], 'supporting_titles': get_supporting_titles(example), 'variants': create_variants(question)}
        records.append(record)
    with OUTPUT_FILE.open('w', encoding='utf-8') as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + '\n')
    pass
    pass
    pass
    pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
