import sys
from transformers import optimization as transformers_optim
from torch.optim import AdamW
from reader_llm import get_reader_llm

# Подменяем AdamW в transformers, чтобы RAGPretrainedModel работал
sys.modules["transformers"].AdamW = AdamW
sys.modules["transformers"].optimization.AdamW = AdamW
from ragatouille import RAGPretrainedModel




def get_reranker(name="colbert-ir/colbertv2.0"):
    name=name
    RERANKER = RAGPretrainedModel.from_pretrained(name)

    return RERANKER


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
    },]
    _ , tokenizer=get_reader_llm()
    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
    prompt_in_chat_format, tokenize=False, add_generation_prompt=True)

    return RAG_PROMPT_TEMPLATE