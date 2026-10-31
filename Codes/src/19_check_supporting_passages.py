import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
metadata_file = DATA / 'passage_metadata.jsonl'
questions_file = DATA / 'prompt_variants.jsonl'
results_file = DATA / 'answer_generation_pilot_v2.csv'
import pandas as pd
with open(metadata_file, 'r', encoding='utf-8') as f:
    passages = [json.loads(line) for line in f if line.strip()]
with open(questions_file, 'r', encoding='utf-8') as f:
    questions = {row['id']: row for row in (json.loads(line) for line in f if line.strip())}
results = pd.read_csv(results_file)
ids = results['question_id'].drop_duplicates().head(5).tolist()
title_to_text = {row.get('title', ''): row.get('text', row.get('passage', row.get('content', ''))) for row in passages}
pass
for qid in ids:
    q = questions.get(qid)
    if not q:
        continue
    support_titles = q.get('supporting_titles', [])
    if not support_titles:
        support_titles = q.get('supporting_facts', [])
    pass
    pass
    pass
    pass
    for title in support_titles:
        if isinstance(title, list):
            title = title[0]
        found = title in title_to_text
        pass
        pass
        if found:
            pass
