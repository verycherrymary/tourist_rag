from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def get_reader_llm(name="Qwen/Qwen2.5-3B-Instruct"):
    READER_MODEL_NAME = name
    
    # Для CPU-only лучше не использовать device_map
    model = AutoModelForCausalLM.from_pretrained(
        READER_MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    
    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        # Убираем device, так как модель уже на CPU
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=50  # Еще больше уменьшаем для надежности
    )
    return READER_LLM