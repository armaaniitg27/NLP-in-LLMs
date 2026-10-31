from pathlib import Path
import pandas as pd
PROJECT_DIR = Path('D:\\Sem 7\\NLP with LLMs\\rag-prompt-robustness')
DATA_DIR = PROJECT_DIR / 'data'
files_to_check = ['RAG_28q_assistant_audit_labels_validated_batch6.csv', 'RAG_prompt_variability_answer_summary.csv', 'RAG_prompt_variability_retrieval_summary.csv', 'full_answer_generation_results_28q.csv']
for filename in files_to_check:
    file_path = DATA_DIR / filename
    pass
    pass
    if not file_path.exists():
        pass
        continue
    try:
        df = pd.read_csv(file_path)
        pass
        pass
        for column in df.columns:
            pass
        pass
        pass
    except Exception as exc:
        pass
