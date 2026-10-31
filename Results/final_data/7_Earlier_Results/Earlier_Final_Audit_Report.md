# Final audit: prompt variability in RAG retrieval and answer quality

## Dataset and review status
- 112 generated answers across 28 question groups and four prompt variants.
- The 24-row sample and remaining 88-row worksheet together cover all 112 answer rows.
- The project author reviewed the annotations and checked the proposed labels against the available question, answer, and context.
- Labels remain judgement-based and exploratory; the labels are based on the available question, generated answer, and retrieved context.

## Correctness by prompt variant
 Variant Fully correct Partially correct Incorrect Mean Recall@5 Complete supporting-title retrieval 
------:---:---:---:---:
 original 8/28 (28.6%) 6/28 (21.4%) 14/28 (50.0%) 69.64% 12/28 (42.9%) \n paraphrase_1 7/28 (25.0%) 3/28 (10.7%) 18/28 (64.3%) 64.29% 9/28 (32.1%) \n paraphrase_2 8/28 (28.6%) 5/28 (17.9%) 15/28 (53.6%) 66.07% 10/28 (35.7%) \n paraphrase_3 6/28 (21.4%) 5/28 (17.9%) 17/28 (60.7%) 66.07% 10/28 (35.7%) \n
## Paired full-correctness comparisons
A paired comparison tests whether each paraphrase changed the number of fully correct answers relative to the original, preserving the fact that the same question groups are evaluated under each prompt. The exact McNemar tests are exploratory and have low power with 28 pairs.

 Comparison Original fully correct Variant fully correct Gained Lost Net change Exact p-value 
------:---:---:---:---:---:
 original_vs_paraphrase_1 8/28 7/28 3 4 -3.57 pp 1.000 \n original_vs_paraphrase_2 8/28 8/28 3 3 +0.00 pp 1.000 \n original_vs_paraphrase_3 8/28 6/28 1 3 -7.14 pp 0.625 \n
## Failure-type distribution
Across all 112 answers:
- Generation failure: 43 (38.4%).
- The needed information was not retrieved: 42 (37.5%).
- Incomplete answer: 7 (6.2%).
- No apparent failure: 20 (17.9%).

These are dominant failure categories assigned in the audit; they are not mutually causal diagnoses and depend on title-level evidence-availability proxies.

## Interpretation
1. **Prompt wording changed retrieval composition and coverage.** Mean Recall@5 is lower for all three paraphrases than for the original in this sample. Complete retrieval of all supporting titles also falls for each paraphrase.
2. **Answer correctness shows a similar descriptive pattern, but not a decisive one.** Fully correct answers are 8/28 for the original, 7/28 for paraphrase 1, 8/28 for paraphrase 2, and 6/28 for paraphrase 3.
3. **Paired tests do not establish a statistically significant correctness difference.** All exact McNemar p-values are above 0.05; the small sample means these results should not be interpreted as evidence that the prompts are equivalent.
4. **Retrieval coverage alone does not explain all answer failures.** Some answers are incorrect despite all supporting article titles appearing in the top five, and some correct answers can be produced with incomplete title-level retrieval. Article titles do not guarantee the answer-bearing sentence was retrieved or used.
5. **Generation and retrieval failures both occur.** The audit assigned 43 cases to generation failure and 42 to retrieval failure. This supports analyzing the two stages separately, but does not prove a causal relationship between a specific retrieval metric and answer correctness.

## Limitations
- Small curated set: 28 question groups; results are exploratory and not broadly generalizable.
- Title-level Recall@5 and complete supporting-title retrieval are proxies, not sentence-level evidence coverage.
- The project author reviewed the annotations against the available question, answer, and context. No separate external-validation study is documented in this package.
- Failure types are judgement-based, and a single dominant label can simplify mixed causes.
- The answer correctness labels assess agreement with reference answers and completeness; they should not automatically be equated with a formal hallucination rate.

## Suggested report-ready conclusion
In this 28-question exploratory evaluation, paraphrasing altered the composition of the retrieved top-five passages and was associated with lower mean title-level Recall@5 for all three paraphrases. Fully correct answer counts also varied, ranging from six to eight out of 28, but paired exact tests did not show statistically significant differences from the original prompt. Audited failures included both missing-evidence retrieval problems and generation errors despite available context. These findings suggest that RAG robustness should be evaluated at both retrieval and answer levels; they do not establish that paraphrasing causally increases hallucination.
