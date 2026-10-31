import json
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
input_path = DATA_DIR / 'hotpotqa_sample.jsonl'
output_path = DATA_DIR / 'passages.jsonl'
passages = {}
examples_processed = 0
with input_path.open('r', encoding='utf-8') as f:
    for line in f:
        example = json.loads(line)
        examples_processed += 1
        context = example['context']
        titles = context['title']
        sentences_list = context['sentences']
        for title, sentences in zip(titles, sentences_list):
            text = ' '.join(sentences).strip()
            if not text:
                continue
            if title not in passages:
                passages[title] = {'title': title, 'text': text}
with output_path.open('w', encoding='utf-8') as f:
    for passage in passages.values():
        f.write(json.dumps(passage, ensure_ascii=False) + '\n')
pass
pass
pass
pass
for passage in list(passages.values())[:3]:
    pass
    pass
