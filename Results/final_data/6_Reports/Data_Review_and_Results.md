# Data audit and results summary

## What was checked
The prompt-variation file has 112 rows: 28 question groups with four versions of each question. The summaries and paired comparisons were recalculated from that file.

## Baseline hallucination labels
The two copies of `Book2_cleaned_hallucination_annotations.csv` disagree on four labels for question `5ab53b48554299488d4d9919`. In the working copy, the two truncated answers are marked `uncertain`; the two paraphrase answers are marked `yes` based on the claims and the retrieved passages available here. Two rows marked as stale prompt results were left out. That leaves 110 usable rows: 76 yes, 32 no and 2 uncertain.

There are two ways to report the yes-rate, depending on the denominator. If uncertain cases are left out, it is 76 out of 108, or 70.37%. If all usable rows are counted, it is 76 out of 110, or 69.09%. The report should state which denominator is being used.

## Rows still needing a human check
The working file still has 52 preliminary annotations and 4 added rows flagged for review. Along with the two truncated answers marked uncertain, that makes 58 rows in the review queue. the project author reports reviewing each annotation; labels marked uncertain remain uncertain where the available answer is incomplete.

## Prompt-variation results
`05_statistics/prompt_variability_summary.csv` gives the counts and rates for each question version. The paired tests compare the original wording with each paraphrase using the same question IDs. Exact McNemar p-values are based on pairs where the outcome changed.

## Comparing the three methods
The recalculated comparison uses the prompt-variation labels for the baseline, the evidence-constrained annotation file, and the query-rewriting annotation file. Baseline hallucination labels come from a separate filtered dataset, so their denominator differs from the 112-row correctness and evidence-support results. The paired comparison excludes uncertain baseline hallucination labels where relevant.

## How to interpret the results
- These are early results from 28 question groups. They do not show that the same pattern will hold for every RAG system or dataset.
- The annotations were prepared with AI assistance and still need independent human validation before making strong claims.
- Older files are kept in `08_reference_outputs/` so earlier decisions can be traced. For this working package, use the recalculated tables in `05_statistics/` after the remaining rows have been checked.
