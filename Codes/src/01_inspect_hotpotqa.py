from datasets import load_dataset
import json
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)
pass
dataset = load_dataset('hotpotqa/hotpot_qa', 'distractor', split='train[:1000]')
pass
pass
pass
example = dataset[0]
pass
pass
pass
pass
pass
output_path = DATA_DIR / 'hotpotqa_sample.jsonl'
with output_path.open('w', encoding='utf-8') as f:
    for row in dataset:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
pass
pass
