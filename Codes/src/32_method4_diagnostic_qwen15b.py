import re
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
INPUT_FILE = DATA_DIR / 'method4_diagnostic_results.csv'
OUTPUT_FILE = DATA_DIR / 'method4_diagnostic_qwen15b.csv'
MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'
MAX_INPUT_TOKENS = 4096
MAX_NEW_TOKENS = 180
SYSTEM_PROMPT = 'You are a careful evidence verifier for a retrieval-augmented\nquestion-answering system.\n\nUse ONLY the supplied retrieved context. Do not use outside knowledge.\nCheck whether the INITIAL ANSWER answers the exact QUESTION.\nDo not transfer a fact about one entity to another entity.\nA passage mentioning a related topic is not enough.\nIf the context does not establish the answer, choose INSUFFICIENT_EVIDENCE.\n\nReturn exactly these four lines:\nLABEL: <one label>\nEVIDENCE: <short exact quote copied from the context, or NONE>\nREASON: <one short sentence>\n\nAllowed labels:\nSUPPORTED - context directly supports the answer to the exact question.\nPARTIALLY_SUPPORTED - context supports only part of the answer.\nCONTRADICTED - context directly conflicts with the answer.\nINSUFFICIENT_EVIDENCE - context does not establish the answer.\n\nDo not choose SUPPORTED merely because the answer sounds plausible.\n'

def normalize(value):
    if pd.isna(value):
        return ''
    return re.sub('\\s+', ' ', str(value)).strip()

def load_input():
    if not INPUT_FILE.exists():
        raise FileNotFoundError()
    df = pd.read_csv(INPUT_FILE).fillna('')
    required = {'diagnostic_case', 'question', 'initial_answer', 'retrieved_context'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError()
    for col in ['question', 'initial_answer', 'retrieved_context']:
        df[col] = df[col].map(normalize)
    return df

def load_model():
    pass
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if torch.cuda.is_available():
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map='auto')
            pass
        except Exception as exc:
            pass
            pass
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        pass
    model.eval()
    return (model, tokenizer)

def classify(model, tokenizer, question, answer, context):
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': f'QUESTION:\n{question}\n\nINITIAL ANSWER:\n{answer}\n\nRETRIEVED CONTEXT:\n{context}\n\nVerify the initial answer now.'}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=MAX_INPUT_TOKENS)
    try:
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except StopIteration:
        pass
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated = output[0][inputs['input_ids'].shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    label_match = re.search('\\b(SUPPORTED|PARTIALLY_SUPPORTED|CONTRADICTED|INSUFFICIENT_EVIDENCE)\\b', raw.upper())
    label = label_match.group(1) if label_match else 'PARSE_ERROR'
    evidence_match = re.search('(?im)^EVIDENCE:\\s*(.*)$', raw)
    reason_match = re.search('(?im)^REASON:\\s*(.*)$', raw)
    evidence = evidence_match.group(1).strip() if evidence_match else ''
    reason = reason_match.group(1).strip() if reason_match else ''
    quote_is_exact = evidence.upper() == 'NONE' or (bool(evidence) and evidence in context)
    return (raw, label, evidence, reason, quote_is_exact)

def main():
    df = load_input()
    model, tokenizer = load_model()
    results = []
    for i, row in df.iterrows():
        pass
        raw, label, evidence, reason, quote_is_exact = classify(model, tokenizer, row['question'], row['initial_answer'], row['retrieved_context'])
        pass
        pass
        pass
        pass
        result = row.to_dict()
        result.update({'model_name': MODEL_NAME, 'model_label': label, 'model_evidence': evidence, 'model_reason': reason, 'raw_model_output': raw, 'quote_is_exact_in_context': quote_is_exact})
        results.append(result)
    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    pass
    pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
