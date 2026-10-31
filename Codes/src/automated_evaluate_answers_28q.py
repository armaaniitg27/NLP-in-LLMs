import ast
import json
import re
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
INPUT_FILE = Path('data/full_answer_generation_results_28q.csv')
OUTPUT_FILE = Path('data/automated_answer_evaluation_28q.csv')
MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
MAX_NEW_TOKENS = 220
MAX_INPUT_LENGTH = 8192
PASSAGE_CHAR_LIMIT = 6500

def parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value) or not str(value).strip():
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    return parsed if isinstance(parsed, list) else []

def normalized(text):
    text = str(text or '').lower()
    text = re.sub('[^a-z0-9]+', ' ', text)
    return ' '.join(text.split())

def deterministic_checks(row):
    answer = str(row.get('generated_answer', '') or '').strip()
    reference = str(row.get('reference_answer', '') or '').strip()
    context = str(row.get('retrieved_context', '') or '')
    supporting = parse_list(row.get('supporting_titles', '[]'))
    retrieved = parse_list(row.get('retrieved_titles', '[]'))
    missing = parse_list(row.get('missing_titles', '[]'))
    retrieved_norm = {normalized(x) for x in retrieved}
    support_norm = {normalized(x) for x in supporting}
    support_retrieved = sorted(support_norm & retrieved_norm)
    support_missing = sorted(support_norm - retrieved_norm)
    answer_norm = normalized(answer)
    reference_norm = normalized(reference)
    exact_ref_match = bool(reference_norm) and (reference_norm in answer_norm or answer_norm in reference_norm)
    return {'det_exact_or_substring_reference_match': exact_ref_match, 'det_supporting_title_count': len(supporting), 'det_supporting_titles_retrieved_count': len(support_retrieved), 'det_supporting_titles_not_in_top5_count': len(support_missing), 'det_supporting_titles_retrieved': json.dumps(support_retrieved, ensure_ascii=False), 'det_supporting_titles_not_in_top5': json.dumps(support_missing, ensure_ascii=False), 'det_missing_passage_count': len(missing), 'det_answer_empty': not bool(answer), 'det_context_characters': len(context)}

def extract_json(text):
    text = text.strip()
    text = re.sub('^```(?:json)?\\s*', '', text, flags=re.I)
    text = re.sub('\\s*```$', '', text)
    start, end = (text.find('{'), text.rfind('}'))
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError()
    df = pd.read_csv(INPUT_FILE)
    required = {'question_id', 'variant', 'query_used', 'reference_answer', 'retrieved_context', 'generated_answer', 'supporting_titles', 'retrieved_titles', 'missing_titles'}
    absent = sorted(required - set(df.columns))
    if absent:
        raise ValueError()
    if len(df) != 112 or df['question_id'].nunique() != 28:
        raise ValueError()
    if df.duplicated(['question_id', 'variant']).any():
        raise ValueError()
    expected = {'original', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3'}
    for qid, group in df.groupby('question_id'):
        if len(group) != 4 or set(group['variant']) != expected:
            raise ValueError()
    existing = pd.read_csv(OUTPUT_FILE) if OUTPUT_FILE.exists() else pd.DataFrame()
    if not existing.empty:
        if not {'question_id', 'variant'}.issubset(existing.columns):
            raise ValueError()
        done = {(str(r.question_id), str(r.variant)) for r in existing[['question_id', 'variant']].itertuples(index=False)}
        result_by_key = {(str(r['question_id']), str(r['variant'])): r for r in existing.to_dict('records')}
    else:
        done, result_by_key = (set(), {})
    pending = []
    for _, row in df.iterrows():
        key = (str(row['question_id']), str(row['variant']))
        if key not in done:
            record = row.to_dict()
            record.update(deterministic_checks(row))
            pending.append((key, record, row))
    if not pending:
        pass
        return
    if not torch.cuda.is_available():
        raise RuntimeError()
    pass
    pass
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, low_cpu_mem_usage=True).to('cuda')
    model.eval()
    system = 'You are a cautious research assistant auditing a retrieval-augmented answer. Treat provided documents as the only evidence for support. The reference answer is a correctness aid, not evidence unless it appears in the documents. Return only one JSON object with keys: correctness, evidence_support, abstention, unsupported_claims, likely_failure_category, confidence, rationale. Allowed correctness: correct, partially_correct, incorrect, unclear. Allowed evidence_support: supported, partially_supported, unsupported, insufficient_evidence, unclear. Allowed abstention: appropriate, unnecessary, not_applicable. unsupported_claims must be a short string. Allowed likely_failure_category: none, retrieval_missing_evidence, generation_ignored_evidence, factual_error, incomplete_answer, unsupported_extra_claim, abstention_issue, unclear. Confidence: low, medium, high. Rationale: 1-3 concise sentences. If evidence is missing or ambiguous, choose unclear/insufficient_evidence rather than guessing. These are preliminary labels.'
    allowed = {'correctness': {'correct', 'partially_correct', 'incorrect', 'unclear'}, 'evidence_support': {'supported', 'partially_supported', 'unsupported', 'insufficient_evidence', 'unclear'}, 'abstention': {'appropriate', 'unnecessary', 'not_applicable'}, 'likely_failure_category': {'none', 'retrieval_missing_evidence', 'generation_ignored_evidence', 'factual_error', 'incomplete_answer', 'unsupported_extra_claim', 'abstention_issue', 'unclear'}, 'confidence': {'low', 'medium', 'high'}}
    for index, (key, record, row) in enumerate(pending, start=1):
        context = str(row['retrieved_context'] or '')
        if len(context) > PASSAGE_CHAR_LIMIT:
            context = context[:PASSAGE_CHAR_LIMIT] + '\n[CONTEXT TRUNCATED FOR JUDGE]'
        obj = {'question': str(row['query_used']), 'reference_answer': str(row['reference_answer']), 'generated_answer': str(row['generated_answer']), 'retrieved_evidence': context}
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': 'Judge correctness separately from evidence support. Audit this answer:\n' + json.dumps(obj, ensure_ascii=False)}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors='pt', truncation=True, max_length=MAX_INPUT_LENGTH).to('cuda')
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.03, pad_token_id=tokenizer.eos_token_id)
        new_tokens = output[0][inputs['input_ids'].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        parsed = extract_json(raw)
        if parsed is None:
            record.update({'judge_correctness': 'parse_error', 'judge_evidence_support': 'parse_error', 'judge_abstention': 'parse_error', 'judge_unsupported_claims': '', 'judge_failure_category': 'unclear', 'judge_confidence': 'low', 'judge_rationale': 'Could not parse valid JSON; review manually.', 'judge_raw_output': raw})
        else:

            def safe(field, fallback):
                value = str(parsed.get(field, fallback)).strip().lower()
                return value if value in allowed[field] else fallback
            record.update({'judge_correctness': safe('correctness', 'unclear'), 'judge_evidence_support': safe('evidence_support', 'unclear'), 'judge_abstention': safe('abstention', 'not_applicable'), 'judge_unsupported_claims': str(parsed.get('unsupported_claims', '')), 'judge_failure_category': safe('likely_failure_category', 'unclear'), 'judge_confidence': safe('confidence', 'low'), 'judge_rationale': str(parsed.get('rationale', '')), 'judge_raw_output': raw})
        result_by_key[key] = record
        pd.DataFrame(result_by_key.values()).to_csv(OUTPUT_FILE, index=False)
        pass
    pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
