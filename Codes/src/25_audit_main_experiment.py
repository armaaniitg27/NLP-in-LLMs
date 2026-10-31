import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
retrieval_path = DATA / 'curated_retrieval_results.csv'
variants_path = DATA / 'prompt_variants.jsonl'
paraphrases_path = DATA / 'curated_30_question_paraphrases.csv'
retrieval = pd.read_csv(retrieval_path)
paraphrases = pd.read_csv(paraphrases_path)
with open(variants_path, 'r', encoding='utf-8') as f:
    original_records = [json.loads(line) for line in f if line.strip()]
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
if 'variant' in retrieval.columns:
    pass
pass
for i, record in enumerate(original_records[:3], start=1):
    pass
    for key in ('id', 'question_id', 'question', 'answer', 'supporting_titles'):
        if key in record:
            pass
pass
for col in ('question_id', 'variant', 'query_used', 'retrieved_titles'):
    if col in retrieval.columns:
        pass
pass
