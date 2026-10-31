import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / 'data' / 'oracle_context_pilot.csv'
df = pd.read_csv(FILE)
for question_id, group in df.groupby('question_id', sort=False):
    row = group.iloc[0]
    pass
    pass
    pass
    pass
    context = str(row['supplied_context'])
    pass
    pass
    pass
    pass
    for _, result in group.iterrows():
        pass
