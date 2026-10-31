import re
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
INPUT_FILE = DATA_DIR / 'answer_verification_revision_28q_v4.csv'
OUTPUT_FILE = DATA_DIR / 'method4_diagnostic_results.csv'
MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
MAX_INPUT_TOKENS = 6000
MAX_NEW_TOKENS = 100
TEST_CASES = [('explicit_hospital_evidence', ['MedStar Washington Hospital Center']), ('explicit_song_evidence', ['Sweet Charity']), ('missing_band_location', ['The Drug in Me Is You', 'Falling in Reverse']), ('irrelevant_birth_order_context', ['Joe Manganiello', 'Kirk Kerkorian']), ('explicit_country_evidence', ['Wenling', 'Xinzheng']), ('magazine_evidence', ['First for Women', 'Jane'])]

def norm(value):
    if pd.isna(value):
        return ''
    return re.sub('\\s+', ' ', str(value)).strip()

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError()
    df = pd.read_csv(INPUT_FILE).fillna('')
    required = {'original_question', 'initial_answer', 'retrieved_context'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError()
    df['original_question'] = df['original_question'].map(norm)
    df['initial_answer'] = df['initial_answer'].map(norm)
    df['retrieved_context'] = df['retrieved_context'].map(norm)
    return df

def select_test_rows(df):
    selected = []
    used_indices = set()
    for case_name, keywords in TEST_CASES:
        mask = pd.Series(True, index=df.index)
        for keyword in keywords:
            mask &= df['original_question'].str.contains(re.escape(keyword), case=False, na=False)
        candidates = df[mask & ~df.index.isin(used_indices)]
        if candidates.empty:
            mask = pd.Series(False, index=df.index)
            for keyword in keywords:
                mask |= df['original_question'].str.contains(re.escape(keyword), case=False, na=False) | df['retrieved_context'].str.contains(re.escape(keyword), case=False, na=False)
            candidates = df[mask & ~df.index.isin(used_indices)]
        if candidates.empty:
            pass
            continue
        row = candidates.iloc[0].copy()
        row['diagnostic_case'] = case_name
        selected.append(row)
        used_indices.add(row.name)
    if not selected:
        raise ValueError()
    return pd.DataFrame(selected).reset_index(drop=True)

def load_model():
    pass
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to('cuda')
        pass
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        pass
    model.eval()
    return (model, tokenizer)

def generate(model, tokenizer, question, answer, context):
    messages = [{'role': 'system', 'content': 'You are checking whether an answer is supported by a supplied passage. Use only the passage. Do not use outside knowledge. Do not transfer facts between different people, places, songs, or organizations. Output exactly one label and nothing else: SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE.'}, {'role': 'user', 'content': f'QUESTION:\n{question}\n\nANSWER TO CHECK:\n{answer}\n\nPASSAGE:\n{context}\n\nChoose one label:\nSUPPORTED = the passage directly supports the answer to this exact question.\nPARTIALLY_SUPPORTED = only part of the answer is supported.\nCONTRADICTED = the passage directly conflicts with the answer.\nINSUFFICIENT_EVIDENCE = the passage does not establish the answer.\nLabel:'}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=MAX_INPUT_TOKENS)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    new_tokens = output[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def main():
    df = load_data()
    tests = select_test_rows(df)
    pass
    for _, row in tests.iterrows():
        pass
    model, tokenizer = load_model()
    results = []
    for _, row in tests.iterrows():
        pass
        raw = generate(model, tokenizer, row['original_question'], row['initial_answer'], row['retrieved_context'])
        pass
        results.append({'diagnostic_case': row['diagnostic_case'], 'question_id': row.get('question_id', ''), 'variant': row.get('variant', ''), 'question': row['original_question'], 'initial_answer': row['initial_answer'], 'retrieved_context': row['retrieved_context'], 'raw_model_output': raw})
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    pass
    pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
