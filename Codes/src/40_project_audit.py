from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
DATA = ROOT / 'data'
OUTPUT_CSV = DATA / 'project_audit_inventory.csv'
FINAL_SCRIPTS = {'03_build_faiss_index.py', '09_run_prompt_experiment.py', '26_generate_full_answers.py', '28_evidence_constrained_generation.py', '29_query_rewriting_generation.py'}
OPTIONAL_FINAL = {'33_reranking_retrieval.py'}
FINAL_RESULTS = {'retrieval_results_28q_rerun.csv', 'retrieval_metrics_28q_rerun.csv', 'retrieval_metrics_28q_rerun_summary.csv', 'rag_prompt_robustness_final_112_answers.csv', 'evidence_constrained_results_28q.csv', 'query_rewriting_results_28q.csv', 'reranking_results_28q.csv', 'reranking_summary_28q.csv', 'reranking_baseline_comparison_28q.csv', 'three_method_summary_28q.csv', 'three_method_by_variant_28q.xlsx', 'three_method_paired_tests_28q.xlsx'}
rows = []
for folder in [SRC, DATA]:
    if not folder.exists():
        continue
    for file in sorted(folder.iterdir()):
        if not file.is_file():
            continue
        category = 'other'
        if folder == SRC:
            category = 'script'
        elif file.suffix == '.csv':
            category = 'csv'
        elif file.suffix == '.xlsx':
            category = 'excel'
        elif file.suffix in {'.jsonl', '.json'}:
            category = 'json'
        elif file.suffix == '.md':
            category = 'markdown'
        elif file.suffix == '.png':
            category = 'image'
        include = False
        reason = ''
        if file.name in FINAL_SCRIPTS:
            include = True
            reason = 'Final experiment script'
        elif file.name in OPTIONAL_FINAL:
            include = True
            reason = 'Final reranking script'
        elif file.name in FINAL_RESULTS:
            include = True
            reason = 'Final result file'
        rows.append({'folder': folder.name, 'filename': file.name, 'size_kb': round(file.stat().st_size / 1024, 1), 'category': category, 'include_final_repo': include, 'reason': reason})
audit = pd.DataFrame(rows)
audit.to_csv(OUTPUT_CSV, index=False)
pass
pass
pass
pass
pass
final = audit[audit.include_final_repo]
pass
pass
for _, r in final.iterrows():
    pass
pass
