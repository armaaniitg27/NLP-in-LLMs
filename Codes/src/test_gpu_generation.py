import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
pass
pass
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda')
question = 'Kim Warwick and Matt Lucena both play what sport?'
prompt = f"Answer the user's question using the information in the documents.\nGive a short, direct answer. Do not deliberately guess when the documents do not provide enough information.\n\nQuestion: {question}\n\nAnswer:"
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=48, do_sample=False)
answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
pass
pass
