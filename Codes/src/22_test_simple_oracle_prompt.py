import json
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / 'data' / 'oracle_context_pilot.csv'
OUTPUT_FILE = ROOT / 'data' / 'oracle_simple_prompt_test.csv'
MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'
df = pd.read_csv(INPUT_FILE)
selected_questions = ['Kim Warwick and Matt Lucena both play what sport?', 'Who was born first, Joe Manganiello or Kirk Kerkorian?', 'What award did a single from the first Foo Fighters album to feature Chris Shiflett win?', 'Seoul Train and Making North America are both films in what style of cinema or film making genre?']
df = df[(df['variant'] == 'original') & df['question'].isin(selected_questions)].copy()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()
results = []
for _, row in df.iterrows():
    question = row['question']
    context = row['supplied_context']
    messages = [{'role': 'system', 'content': "Answer the user's question using the information in the documents. Give a short, direct answer."}, {'role': 'user', 'content': f'Question: {question}\n\nDocuments:\n{context}\n\nAnswer:'}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=4096)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    new_tokens = output[:, inputs['input_ids'].shape[1]:]
    answer = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
    results.append({'question': question, 'reference_answer': row['reference_answer'], 'generated_answer': answer, 'supplied_context': context})
    pass
    pass
pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
pass
