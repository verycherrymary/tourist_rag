from langchain_community.embeddings import HuggingFaceEmbeddings
from loader import load_and_split_markdown
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy



def get_retriever(name='intfloat/multilingual-e5-large'):
    embedding_model = HuggingFaceEmbeddings(
    model_name=name,
    multi_process=True,
    # model_kwargs={"device": "cuda"},
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # `True` для косинусного сходства
     )
    docs_processed=load_and_split_markdown()
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
    docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE )

    return embedding_model, KNOWLEDGE_VECTOR_DATABASE 