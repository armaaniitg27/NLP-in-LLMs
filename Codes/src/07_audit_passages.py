import json
from collections import defaultdict
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
input_path = DATA_DIR / 'hotpotqa_sample.jsonl'
title_texts = defaultdict(set)
title_counts = defaultdict(int)
with input_path.open('r', encoding='utf-8') as f:
    for line in f:
        example = json.loads(line)
        context = example['context']
        for title, sentences in zip(context['title'], context['sentences']):
            title_counts[title] += 1
            text = ' '.join(sentences).strip()
            title_texts[title].add(text)
unique_titles = len(title_counts)
repeated_titles = sum((count > 1 for count in title_counts.values()))
titles_with_different_text = {title: texts for title, texts in title_texts.items() if len(texts) > 1}
pass
pass
pass
pass
pass
pass
for title, texts in list(titles_with_different_text.items())[:5]:
    pass
    pass
    for i, text in enumerate(list(texts)[:2], start=1):
        pass
