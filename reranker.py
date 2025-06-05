import os
os.environ["NO_CUDA_EXT"] = "1"  # Полностью отключаем C++ расширения

from typing import Optional
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

def get_reranker(name: Optional[str] = None) -> CrossEncoder:
    """
    Инициализация ранкера с использованием CrossEncoder 
    """
    # Используем более легкую модель по умолчанию
    model_name = name or 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    return CrossEncoder(model_name)

def get_rag_prompt_template():
    prompt_in_chat_format = [
        {
            "role": "system",
        "content": """Используй информацию из контекста, чтобы дать полный ответ на вопрос.
    Отвечай только на заданный вопрос, ответ должен быть чётким и соответствующим вопросу.
    Указывай номер исходного документа, когда это уместно.
    Если ответ нельзя вывести из контекста, дай ответ,который знаешь, но обязательно напиши,что ответ дан не из контекста.
    Отвечай строго на русском языке, даже если контекст содержит текст на других языках.""",  # Добавлено требование русского языка
    },
    {
        "role": "user",
        "content": """Контекст:
   {context}
   ---
   Вот вопрос, на который нужно ответить.

   Вопрос: {question}""", 
        }
    ]
    READER_MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    return tokenizer.apply_chat_template(
        prompt_in_chat_format, 
        tokenize=False, 
        add_generation_prompt=True
    )