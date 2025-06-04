from transformers import Pipeline
from langchain.vectorstores import FAISS
from reranker import get_reranker, get_rag_prompt_template
from typing import List, Tuple
from langchain.docstore.document import Document as LangchainDocument
import streamlit as st  # Добавляем импорт Streamlit

def answer_with_rag2(
    question: str,
    llm: Pipeline,
    knowledge_index: FAISS,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 10,
) -> Tuple[str, List[LangchainDocument]]:
    # Собираем документы с помощью ретривера
    st.write("=> Retrieving documents...")
    relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    
    # Сохраняем полные документы с метаданными
    full_docs = relevant_docs.copy()
    relevant_contents = [doc.page_content for doc in relevant_docs]

    reranker = get_reranker()
    if reranker:
        st.write("=> Reranking documents...")
        try:
            reranked_results = reranker.rerank(question, relevant_contents, k=num_docs_final)
            if isinstance(reranked_results[0], dict) and "index" in reranked_results[0]:
                full_docs = [full_docs[result["index"]] for result in reranked_results]
                relevant_contents = [result["content"] for result in reranked_results]
            else:
                reranked_contents = [result["content"] if isinstance(result, dict) else result for result in reranked_results]
                full_docs = [doc for doc in full_docs if doc.page_content in reranked_contents][:num_docs_final]
                relevant_contents = reranked_contents[:num_docs_final]
        except Exception as e:
            st.error(f"Ошибка при реранкинге: {e}")
            full_docs = full_docs[:num_docs_final]
            relevant_contents = relevant_contents[:num_docs_final]

    # Формируем контекст для промпта
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {i}:::\n{doc}\n" for i, doc in enumerate(relevant_contents[:num_docs_final])])

    # Генерируем ответ
    st.write("=> Generating answer...")
    RAG_PROMPT_TEMPLATE = get_rag_prompt_template()
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)
    answer = llm(final_prompt)[0]["generated_text"]

    # Отображаем результаты с изображениями и координатами
    st.markdown("\n## Ответ")
    st.write(answer)
    
    st.markdown("## Использованные источники")
    for i, doc in enumerate(full_docs[:num_docs_final]):
        with st.expander(f"Документ {i+1}"):
            st.write(doc.page_content)
            
            # Отображаем координаты
            if hasattr(doc, 'metadata') and doc.metadata:
                if "longitude" in doc.metadata and "latitude" in doc.metadata:
                    st.write(f"📍 Широта: {doc.metadata['latitude']}, Долгота: {doc.metadata['longitude']}")
                
                # Отображаем изображение
                if "image" in doc.metadata and doc.metadata["image"]:
                    try:
                        if isinstance(doc.metadata["image"], str):
                            if doc.metadata["image"].startswith('data:image'):
                                # Обработка base64 изображений
                                st.image(doc.metadata["image"], caption=f"Изображение из документа {i+1}")
                            else:
                                # Предполагаем, что это путь к файлу
                                st.image(doc.metadata["image"], caption=f"Изображение из документа {i+1}")
                        elif isinstance(doc.metadata["image"], bytes):
                            # Обработка бинарных данных изображения
                            st.image(doc.metadata["image"], caption=f"Изображение из документа {i+1}")
                    except Exception as e:
                        st.error(f"Ошибка загрузки изображения: {e}")

    return answer, full_docs[:num_docs_final]