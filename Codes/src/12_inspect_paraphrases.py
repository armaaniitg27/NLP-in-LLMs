import json
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / 'data' / 'paraphrase_candidates.jsonl'
with DATA_FILE.open('r', encoding='utf-8') as file:
    records = [json.loads(line) for line in file]
candidate_counts = [len(record['paraphrases']) for record in records]
flagged = [(record, item) for record in records for item in record['paraphrases'] if item['review_needed']]
pass
pass
pass
pass
pass
pass
pass
shown = 0
for record in records:
    if shown >= 10:
        break
    if not record['paraphrases']:
        continue
    pass
    pass
    pass
    pass
    for i, item in enumerate(record['paraphrases'], start=1):
        pass
        if item['review_needed']:
            pass
    shown += 1
