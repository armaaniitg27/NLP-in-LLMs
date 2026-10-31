# RAG Prompt Variability — Revised Audit After Targeted Review

## Scope and status

This revision applies targeted evidence-review recommendations to the 112-row answer audit. Eight question groups were inspected in detail; updates were made only to the explicitly checked variants. These are the project author reviewed the annotations against the available evidence. Labels outside the selected review set retain their recorded values..

## Revised correctness counts (n=28 per variant)

 Variant Fully correct Partially correct Incorrect 
------:---:---:
 Original 8/28 (28.6%) 6/28 (21.4%) 14/28 (50.0%) 
 P1 7/28 (25.0%) 3/28 (10.7%) 18/28 (64.3%) 
 P2 8/28 (28.6%) 5/28 (17.9%) 15/28 (53.6%) 
 P3 6/28 (21.4%) 5/28 (17.9%) 17/28 (60.7%) 

## Paired fully-correct analysis

The primary paired endpoint is `final_correctness == correct`; partially correct, incorrect, and unclear labels count as not fully correct. The exact two-sided McNemar/binomial test is applied to discordant paired outcomes.

 Comparison Original correct Variant correct Gained Lost Net change (pp) Exact p-value 
------:---:---:---:---:---:
 Original vs P1 8/28 7/28 3 4 -3.6 1.000 
 Original vs P2 8/28 8/28 3 3 +0.0 1.000 
 Original vs P3 8/28 6/28 1 3 -7.1 0.625 

No p-value should be interpreted as proof of equivalence. The sample is small and curated, so results are exploratory and have limited statistical power.

## Label updates from targeted review

- `5a78e864554299029c4b5ed7` / `original`: correctness `partly_correct` → `partly_correct`; evidence `partly_supported` → `partly_supported`; failure type `incomplete_answer` → `answer_generation_error`.
- `5a7bbc77554299294a54aaaa` / `original`: correctness `correct` → `correct`; evidence `partly_supported` → `partly_supported`; failure type `retrieval_missed_needed_info` → `answer_generation_error`.
- `5a8ce61555429941ae14df36` / `original`: correctness `incorrect` → `incorrect`; evidence `contradicted_by_context` → `insufficient_evidence`; failure type `retrieval_missed_needed_info` → `retrieval_missed_needed_info`.
- `5a8ce61555429941ae14df36` / `paraphrase_1`: correctness `unclear` → `incorrect`; evidence `insufficient_evidence` → `insufficient_evidence`; failure type `retrieval_missed_needed_info` → `retrieval_missed_needed_info`.
- `5a8ce61555429941ae14df36` / `paraphrase_2`: correctness `incorrect` → `incorrect`; evidence `contradicted_by_context` → `unsupported`; failure type `answer_generation_error` → `answer_generation_error`.
- `5ab6ad2855429953192ad35e` / `paraphrase_1`: correctness `incorrect` → `incorrect`; evidence `contradicted_by_context` → `contradicted_by_context`; failure type `answer_generation_error` → `retrieval_missed_needed_info`.

## Interpretation and limitations

- Correctness against a reference answer and support from the retrieved context are separate dimensions. An answer can be factually correct but only partially supported by the passages provided to the model.
- When the relevant supporting article is missing, an incorrect answer should not automatically be described as contradicted_by_context by the retrieved context. Use an insufficient-evidence/retrieval-limited label where appropriate.
- A title-level Recall@5 score does not guarantee that the exact supporting sentence was retrieved.
- This targeted review is selective, and the labels remain temporary until a human checks the question, reference answer, generated answer, and all retrieved passages.
- The findings describe this 28-question curated sample; they do not establish a general causal effect of paraphrasing or a general hallucination rate.

## Recommended next step

Human-check a stratified set of examples from every variant and each correctness/evidence-support category, adjudicate disagreements, then freeze the label definitions before writing the final Results and Discussion. Do not silently treat the assistant audit as human validation.
