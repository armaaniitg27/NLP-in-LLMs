import numpy as np
import pandas as pd
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
INPUT_FILE = DATA_DIR / 'prompt_retrieval_results.csv'
OUTPUT_FILE = DATA_DIR / 'prompt_statistical_analysis.csv'
METRICS = ['recall_at_5', 'complete_support_retrieval', 'reciprocal_rank_at_5', 'top5_overlap_with_original']
BOOTSTRAP_SAMPLES = 10000
RANDOM_SEED = 42

def bootstrap_ci(values, rng):
    values = np.asarray(values, dtype=float)
    n = len(values)
    means = np.empty(BOOTSTRAP_SAMPLES)
    for i in range(BOOTSTRAP_SAMPLES):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    lower, upper = np.percentile(means, [2.5, 97.5])
    return (values.mean(), lower, upper)

def main():
    df = pd.read_csv(INPUT_FILE)
    rng = np.random.default_rng(RANDOM_SEED)
    if df.duplicated(['id', 'variant']).any():
        raise ValueError()
    output = []
    for metric in METRICS:
        pivot = df.pivot(index='id', columns='variant', values=metric)
        if 'original' not in pivot.columns:
            raise ValueError()
        for variant in pivot.columns:
            if variant == 'original':
                continue
            paired_difference = (pivot[variant] - pivot['original']).dropna()
            mean_diff, ci_low, ci_high = bootstrap_ci(paired_difference.to_numpy(), rng)
            output.append({'metric': metric, 'variant': variant, 'questions_compared': len(paired_difference), 'mean_difference_vs_original': mean_diff, 'ci_95_lower': ci_low, 'ci_95_upper': ci_high, 'ci_excludes_zero': ci_low > 0 or ci_high < 0})
    result = pd.DataFrame(output)
    result.to_csv(OUTPUT_FILE, index=False)
    pass
    pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
