import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'
PROMPT = 'Answer the question using the document.\n\nQuestion: Kim Warwick and Matt Lucena both play what sport?\n\nDocument 1: Kim Warwick was an Australian former professional tennis player.\nDocument 2: Matt Lucena is a former professional tennis player.\n\nAnswer:'

def generate_once(model, tokenizer):
    messages = [{'role': 'system', 'content': 'Answer the question using the supplied document. Be concise.'}, {'role': 'user', 'content': PROMPT}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt')
    start = time.time()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    elapsed = time.time() - start
    new_tokens = output[0][inputs['input_ids'].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return (elapsed, answer)

def main():
    pass
    pass
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pass
    model_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, low_cpu_mem_usage=True).to('cpu')
    model_fp16.eval()
    pass
    time_fp16, answer_fp16 = generate_once(model_fp16, tokenizer)
    pass
    pass
    del model_fp16
    import gc
    gc.collect()
    pass
    model_fp32 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32, low_cpu_mem_usage=True).to('cpu')
    model_fp32.eval()
    pass
    model_int8 = torch.ao.quantization.quantize_dynamic(model_fp32, {torch.nn.Linear}, dtype=torch.qint8)
    model_int8.eval()
    del model_fp32
    gc.collect()
    pass
    time_int8, answer_int8 = generate_once(model_int8, tokenizer)
    pass
    pass
    pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
