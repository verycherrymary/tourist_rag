import os
os.environ["NO_CUDA_EXT"] = "1"  # Полностью отключаем C++ расширения

from typing import Optional
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

def get_reranker(name: Optional[str] = None) -> CrossEncoder:
    """
    Инициализация ранкера с использованием CrossEncoder вместо ColBERT
    """
    # Используем более легкую модель по умолчанию
    model_name = name or 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    return CrossEncoder(model_name)

def get_rag_prompt_template():
    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Using the information contained in the context,
            give a comprehensive answer to the question.
            Respond only to the question asked, response should be concise and relevant to the question.
            Provide the number of the source document when relevant.
            If the answer cannot be deduced from the context, do not give an answer.""",
        },
        {
            "role": "user",
            "content": """Context:
{context}
---
Now here is the question you need to answer.

Question: {question}""",
        }
    ]
    READER_MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    return tokenizer.apply_chat_template(
        prompt_in_chat_format, 
        tokenize=False, 
        add_generation_prompt=True
    )