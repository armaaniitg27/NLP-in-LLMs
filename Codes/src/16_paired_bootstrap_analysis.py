from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
INPUT_FILE = DATA / 'curated_retrieval_results.csv'
OUTPUT_FILE = DATA / 'paired_bootstrap_results.csv'
SUMMARY_FILE = DATA / 'paired_bootstrap_summary.csv'
N_BOOTSTRAPS = 10000
SEED = 42
METRICS = {'recall_at_5': 'Recall@5', 'complete_support_retrieval': 'Complete retrieval rate', 'reciprocal_rank_at_5': 'MRR@5'}
VARIANTS = ['paraphrase_1', 'paraphrase_2', 'paraphrase_3']

def bootstrap_mean_ci(values, rng):
    """Bootstrap a mean and return its 95% percentile interval."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    samples = rng.choice(values, size=(N_BOOTSTRAPS, n), replace=True)
    means = samples.mean(axis=1)
    return (float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))

def main():
    results = pd.read_csv(INPUT_FILE)
    required = {'question_id', 'variant', *METRICS.keys(), 'top5_overlap_with_original'}
    missing = required - set(results.columns)
    if missing:
        raise ValueError()
    if results.duplicated(['question_id', 'variant']).any():
        raise ValueError()
    pivoted = results.pivot(index='question_id', columns='variant', values=list(METRICS.keys()))
    expected_variants = {'original', *VARIANTS}
    available_variants = set(results['variant'].unique())
    if not expected_variants.issubset(available_variants):
        raise ValueError()
    if pivoted.isna().any().any():
        raise ValueError()
    rng = np.random.default_rng(SEED)
    comparison_rows = []
    summary_rows = []
    for metric, metric_label in METRICS.items():
        metric_data = pivoted[metric]
        original = metric_data['original'].to_numpy(dtype=float)
        for variant in VARIANTS:
            values = metric_data[variant].to_numpy(dtype=float)
            differences = values - original
            mean_diff, ci_low, ci_high = bootstrap_mean_ci(differences, rng)
            summary_rows.append({'metric': metric, 'metric_label': metric_label, 'variant': variant, 'n_questions': len(values), 'variant_mean': float(values.mean()), 'original_mean': float(original.mean()), 'mean_difference': mean_diff, 'ci_95_lower': ci_low, 'ci_95_upper': ci_high, 'ci_excludes_zero': ci_high < 0 or ci_low > 0})
        for variant in VARIANTS:
            values = metric_data[variant].to_numpy(dtype=float)
            differences = values - original
            n = len(differences)
            sampled_indices = rng.integers(0, n, size=(N_BOOTSTRAPS, n))
            bootstrap_differences = differences[sampled_indices].mean(axis=1)
            comparison_rows.append({'metric': metric, 'metric_label': metric_label, 'variant': variant, 'n_questions': n, 'mean_difference_vs_original': float(differences.mean()), 'ci_95_lower': float(np.quantile(bootstrap_differences, 0.025)), 'ci_95_upper': float(np.quantile(bootstrap_differences, 0.975))})
    for variant in VARIANTS:
        values = results.loc[results['variant'] == variant, 'top5_overlap_with_original'].to_numpy(dtype=float)
        mean, low, high = bootstrap_mean_ci(values, rng)
        comparison_rows.append({'metric': 'top5_overlap_with_original', 'metric_label': 'Top-five overlap (descriptive)', 'variant': variant, 'n_questions': len(values), 'mean_difference_vs_original': np.nan, 'ci_95_lower': low, 'ci_95_upper': high})
    summary = pd.DataFrame(summary_rows)
    comparisons = pd.DataFrame(comparison_rows)
    summary.to_csv(SUMMARY_FILE, index=False)
    comparisons.to_csv(OUTPUT_FILE, index=False)
    pass
    pass
    pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
