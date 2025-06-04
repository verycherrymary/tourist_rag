from reader_llm import get_reader_llm
from retrieval import get_retriever
from answer_rag import answer_with_rag2
import streamlit as st

# Настройка страницы
st.set_page_config(page_title="RAG", layout="wide")
st.title("Туристический путеводитель")
st.header("Города: Ярославль, Екатеринбург, Нижний Новгород, Владимир")

@st.cache_resource
def load_models():
    READER_LLM = get_reader_llm()
    embedding_model, KNOWLEDGE_VECTOR_DATABASE = get_retriever()
    return READER_LLM, embedding_model, KNOWLEDGE_VECTOR_DATABASE

READER_LLM, _, KNOWLEDGE_VECTOR_DATABASE = load_models()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Задайте Ваш вопрос"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Ищу информацию..."):
            answer, sources = answer_with_rag2(
                question=prompt,
                llm=READER_LLM,
                knowledge_index=KNOWLEDGE_VECTOR_DATABASE
            )
            st.markdown(answer)
            
            if sources:
                st.markdown("**Источники информации:**")
                for i, doc in enumerate(sources):
                    with st.expander(f"Источник {i+1}"):
                        st.write(doc.page_content)
                        if hasattr(doc, 'metadata'):
                            if "latitude" in doc.metadata and "longitude" in doc.metadata:
                                st.write(f"📍 Координаты: {doc.metadata['latitude']}, {doc.metadata['longitude']}")
                            if "image" in doc.metadata and doc.metadata["image"]:
                                try:
                                    st.image(doc.metadata["image"], caption=f"Изображение {i+1}")
                                except Exception as e:
                                    st.error(f"Ошибка загрузки изображения: {e}")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})