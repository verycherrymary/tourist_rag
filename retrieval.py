from langchain_community.embeddings import HuggingFaceEmbeddings
from loader import load_and_split_markdown
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()  # Отключает прогресс-бары загрузки


def get_retriever(name='intfloat/multilingual-e5-large'):
    # Убираем multi_process для Windows
    embedding_model = HuggingFaceEmbeddings(
        model_name=name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 4  # Уменьшаем batch_size для CPU
        }
    )
    docs_processed=load_and_split_markdown()
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
    documents=docs_processed, embedding=embedding_model, distance_strategy=DistanceStrategy.COSINE )

    return embedding_model, KNOWLEDGE_VECTOR_DATABASE 