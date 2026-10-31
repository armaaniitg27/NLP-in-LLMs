import json
import re
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
BASELINE_FILE = DATA_DIR / 'rag_prompt_robustness_final_112_answers.csv'
ANNOTATION_FILE = DATA_DIR / 'rag_prompt_robustness_combined_annotations.csv'
OUTPUT_FILE = DATA_DIR / 'answer_verification_revision_28q_v4.csv'
MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
MAX_INPUT_TOKENS = 6000
MAX_NEW_TOKENS = 220
ABSTENTION_TEXT = 'Insufficient evidence in the retrieved context to answer.'
VERIFICATION_LABELS = {'SUPPORTED', 'PARTIALLY_SUPPORTED', 'CONTRADICTED', 'INSUFFICIENT_EVIDENCE'}
REVISION_ACTIONS = {'KEEP', 'REVISE', 'ABSTAIN'}

def normalize_text(value):
    if pd.isna(value):
        return ''
    return str(value).strip()

def normalize_for_match(value):
    return re.sub('\\s+', ' ', normalize_text(value)).strip()

def quote_is_in_context(quote, context):
    quote_norm = normalize_for_match(quote)
    context_norm = normalize_for_match(context)
    return bool(quote_norm) and quote_norm in context_norm

def load_and_match_data():
    baseline = pd.read_csv(BASELINE_FILE)
    annotations = pd.read_csv(ANNOTATION_FILE)
    baseline_required = {'question_id', 'variant', 'original_question', 'reference_answer', 'generated_answer'}
    annotation_required = {'question_id', 'variant', 'generated_answer', 'retrieved_context'}
    if not baseline_required.issubset(baseline.columns):
        raise ValueError()
    if not annotation_required.issubset(annotations.columns):
        raise ValueError()
    baseline['generated_answer'] = baseline['generated_answer'].map(normalize_text)
    annotations['generated_answer'] = annotations['generated_answer'].map(normalize_text)
    baseline['_answer_key'] = baseline['generated_answer']
    annotations['_answer_key'] = annotations['generated_answer']
    evidence = annotations[['question_id', 'variant', '_answer_key', 'retrieved_context']].copy().drop_duplicates()
    if evidence.duplicated(['question_id', 'variant', '_answer_key']).any():
        raise ValueError()
    matched = baseline.merge(evidence, on=['question_id', 'variant', '_answer_key'], how='left', validate='one_to_one')
    if len(matched) != len(baseline):
        raise ValueError()
    if matched['retrieved_context'].isna().any():
        raise ValueError()
    matched['retrieved_context'] = matched['retrieved_context'].map(normalize_text)
    if (matched['retrieved_context'] == '').any():
        raise ValueError()
    if len(matched) != 112:
        raise ValueError()
    if matched.duplicated(['question_id', 'variant']).any():
        raise ValueError()
    pass
    pass
    pass
    return matched

def generate_text(model, tokenizer, messages, max_new_tokens):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=MAX_INPUT_TOKENS)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    new_tokens = output[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def parse_json_response(response):
    text = response.strip()
    text = re.sub('^\\s*```(?:json)?\\s*', '', text, flags=re.IGNORECASE)
    text = re.sub('\\s*```\\s*$', '', text)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return (result, '')
    except json.JSONDecodeError:
        pass
    start = text.find('{')
    if start >= 0:
        try:
            result, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(result, dict):
                return (result, '')
        except json.JSONDecodeError:
            pass
    return ({}, 'INVALID_JSON')

def verify_answer(model, tokenizer, row):
    question = normalize_text(row['original_question'])
    answer = normalize_text(row['generated_answer'])
    context = normalize_text(row['retrieved_context'])
    messages = [{'role': 'system', 'content': 'You are a conservative evidence auditor. Use only the provided context, not outside knowledge. The reference answer is hidden. Evaluate whether the initial answer answers the exact question and whether the context supports it. Do not transfer facts between different entities. Return one valid JSON object only. For assessment, select exactly one allowed label, never the list of labels.'}, {'role': 'user', 'content': f'\nQUESTION:\n{question}\n\nINITIAL ANSWER:\n{answer}\n\nRETRIEVED CONTEXT:\n{context}\n\nChoose exactly ONE assessment value:\n\nSUPPORTED\nPARTIALLY_SUPPORTED\nCONTRADICTED\nINSUFFICIENT_EVIDENCE\n\nDefinitions:\n- SUPPORTED: the answer directly answers the question and its material\n  claims are supported by the context.\n- PARTIALLY_SUPPORTED: some relevant claims are supported, but the answer\n  is incomplete or includes an unsupported material claim.\n- CONTRADICTED: the context directly conflicts with the answer.\n- INSUFFICIENT_EVIDENCE: the context does not establish the answer.\n\nDo not transfer a date, location, identity, or fact from one entity to\nanother. A related document is not enough unless it supports the exact\nanswer. Evidence quotes must be exact excerpts from the context. Use an\nempty list if no relevant quote exists.\n\nExample of the required JSON structure (choose the correct assessment\nfor this row; do not copy the example label automatically):\n{{\n  "assessment": "INSUFFICIENT_EVIDENCE",\n  "reason": "The context does not establish the requested fact.",\n  "evidence_quotes": []\n}}\n'}]
    raw = generate_text(model, tokenizer, messages, MAX_NEW_TOKENS)
    result, error = parse_json_response(raw)
    if error:
        return {'verification_assessment': 'PARSE_ERROR', 'verification_reason': '', 'evidence_quotes': '[]', 'verification_raw_output': raw, 'verification_error': error, 'quote_validation': 'NOT_CHECKED'}
    assessment = normalize_text(result.get('assessment', '')).upper()
    reason = normalize_text(result.get('reason', ''))
    quotes = result.get('evidence_quotes', [])
    if assessment not in VERIFICATION_LABELS:
        return {'verification_assessment': 'PARSE_ERROR', 'verification_reason': reason, 'evidence_quotes': json.dumps(quotes, ensure_ascii=False), 'verification_raw_output': raw, 'verification_error': 'INVALID_ASSESSMENT_LABEL', 'quote_validation': 'NOT_CHECKED'}
    if not isinstance(quotes, list) or not all((isinstance(quote, str) for quote in quotes)):
        return {'verification_assessment': 'PARSE_ERROR', 'verification_reason': reason, 'evidence_quotes': '[]', 'verification_raw_output': raw, 'verification_error': 'INVALID_EVIDENCE_QUOTES', 'quote_validation': 'NOT_CHECKED'}
    if not reason:
        return {'verification_assessment': 'PARSE_ERROR', 'verification_reason': '', 'evidence_quotes': json.dumps(quotes, ensure_ascii=False), 'verification_raw_output': raw, 'verification_error': 'MISSING_REASON', 'quote_validation': 'NOT_CHECKED'}
    invalid_quotes = [quote for quote in quotes if not quote_is_in_context(quote, context)]
    quote_validation = 'PASS' if not invalid_quotes else 'FAIL'
    verification_error = '' if not invalid_quotes else 'QUOTE_NOT_FOUND_IN_CONTEXT'
    if assessment == 'SUPPORTED' and (not quotes or invalid_quotes):
        assessment = 'INSUFFICIENT_EVIDENCE'
        verification_error = verification_error or 'SUPPORTED_WITHOUT_VERIFIABLE_QUOTE'
    return {'verification_assessment': assessment, 'verification_reason': reason, 'evidence_quotes': json.dumps(quotes, ensure_ascii=False), 'verification_raw_output': raw, 'verification_error': verification_error, 'quote_validation': quote_validation}

def make_revision(action, answer, reason, raw='', error='', quote='', quote_validation='NOT_REQUIRED'):
    return {'revision_action': action, 'revised_answer': answer, 'revision_reason': reason, 'revision_raw_output': raw, 'revision_error': error, 'revision_evidence_quote': quote, 'revision_quote_validation': quote_validation}

def revise_answer(model, tokenizer, row, verification):
    question = normalize_text(row['original_question'])
    initial_answer = normalize_text(row['generated_answer'])
    context = normalize_text(row['retrieved_context'])
    assessment = verification['verification_assessment']
    if assessment == 'PARSE_ERROR':
        return make_revision('ABSTAIN', ABSTENTION_TEXT, 'Verification output was invalid; conservatively abstained.', error='VERIFICATION_FAILED')
    if assessment == 'SUPPORTED' and (not verification['verification_error']):
        return make_revision('KEEP', initial_answer, 'The answer was marked supported and its evidence quote passed the exact-context check. The initial answer was kept.')
    if assessment == 'INSUFFICIENT_EVIDENCE':
        return make_revision('ABSTAIN', ABSTENTION_TEXT, 'The retrieved context does not establish a reliable answer.')
    messages = [{'role': 'system', 'content': 'You are a conservative answer corrector. Use only the provided context. Never use outside knowledge or transfer facts between entities. Propose a correction only when the context directly supports the answer to the exact question. Otherwise abstain. Return one valid JSON object only. For action, select exactly one allowed value.'}, {'role': 'user', 'content': f'''\nQUESTION:\n{question}\n\nINITIAL ANSWER:\n{initial_answer}\n\nVERIFICATION ASSESSMENT:\n{assessment}\n\nRETRIEVED CONTEXT:\n{context}\n\nChoose exactly ONE action:\nREVISE\nABSTAIN\n\nRules:\n- REVISE only if the context directly supports a corrected answer to this\n  exact question.\n- Do not use a different entity's date, location, identity, or other fact.\n- Do not revise merely to change wording.\n- A REVISE answer must have a short, exact supporting quote copied from\n  the context.\n- If no exact quote supports the proposed answer, choose ABSTAIN.\n- For ABSTAIN, revised_answer must be exactly:\n  "{ABSTENTION_TEXT}"\n\nExample structure (select the appropriate action for this row):\n{{\n  "action": "ABSTAIN",\n  "revised_answer": "{ABSTENTION_TEXT}",\n  "reason": "The context does not establish a reliable answer.",\n  "evidence_quote": ""\n}}\n'''}]
    raw = generate_text(model, tokenizer, messages, MAX_NEW_TOKENS)
    result, error = parse_json_response(raw)
    if error:
        return make_revision('ABSTAIN', ABSTENTION_TEXT, 'Revision output was invalid; conservatively abstained.', raw=raw, error=error)
    action = normalize_text(result.get('action', '')).upper()
    revised_answer = normalize_text(result.get('revised_answer', ''))
    reason = normalize_text(result.get('reason', ''))
    evidence_quote = result.get('evidence_quote', '')
    if not isinstance(evidence_quote, str):
        evidence_quote = ''
    if action not in REVISION_ACTIONS:
        return make_revision('ABSTAIN', ABSTENTION_TEXT, reason or 'Invalid revision action; conservatively abstained.', raw=raw, error='INVALID_ACTION')
    if not reason:
        return make_revision('ABSTAIN', ABSTENTION_TEXT, 'Missing revision reason; conservatively abstained.', raw=raw, error='MISSING_REVISION_REASON')
    if action == 'ABSTAIN':
        return make_revision('ABSTAIN', ABSTENTION_TEXT, reason, raw=raw)
    if not revised_answer:
        return make_revision('ABSTAIN', ABSTENTION_TEXT, 'The proposed correction was empty; conservatively abstained.', raw=raw, error='MISSING_REVISED_ANSWER')
    if not evidence_quote or not quote_is_in_context(evidence_quote, context):
        return make_revision('ABSTAIN', ABSTENTION_TEXT, 'The proposed correction lacked a verifiable exact evidence quote; conservatively abstained.', raw=raw, error='REVISION_QUOTE_NOT_FOUND_IN_CONTEXT', quote=evidence_quote, quote_validation='FAIL')
    return make_revision('REVISE', revised_answer, reason, raw=raw, quote=evidence_quote, quote_validation='PASS')

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

def main():
    df = load_and_match_data()
    model, tokenizer = load_model()
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Verifying and revising answers (v4)'):
        verification = verify_answer(model, tokenizer, row)
        revision = revise_answer(model, tokenizer, row, verification)
        results.append({'question_id': row['question_id'], 'variant': row['variant'], 'original_question': row['original_question'], 'reference_answer': row['reference_answer'], 'initial_answer': row['generated_answer'], 'retrieved_context': row['retrieved_context'], **verification, **revision, 'model_name': MODEL_NAME})
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    pass
    pass
    pass
    for column, title in [('verification_assessment', 'Verification assessments'), ('revision_action', 'Revision actions'), ('verification_error', 'Verification errors'), ('revision_error', 'Revision errors'), ('quote_validation', 'Verification quote validation'), ('revision_quote_validation', 'Revision quote validation')]:
        pass
        values = results_df[column].fillna('').replace('', 'NONE').value_counts(dropna=False)
        pass
    pass
    pass
if __name__ == '__main__':
    main()
