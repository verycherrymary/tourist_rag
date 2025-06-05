import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NO_CUDA_EXT"] = "1"


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
            
            # if sources:
            #     st.markdown("**Источники информации:**")
            #     for i, doc in enumerate(sources):
            #         with st.expander(f"Источник {i+1}"):
            #             st.write(doc.page_content)
            #             if hasattr(doc, 'metadata'):
            #                 if "latitude" in doc.metadata and "longitude" in doc.metadata:
            #                     st.write(f"📍 Координаты: {doc.metadata['latitude']}, {doc.metadata['longitude']}")
            #                 if "image" in doc.metadata and doc.metadata["image"]:
            #                     try:
            #                         if isinstance(doc.metadata["image"], str):
            #                             if doc.metadata["image"].startswith('/9j/'):
            #                                 import base64
            #                                 from io import BytesIO
            #                                 from PIL import Image
                                            
            #                                 img_bytes = base64.b64decode(doc.metadata["image"])
            #                                 img = Image.open(BytesIO(img_bytes))
            #                                 st.image(img, caption=f"Изображение {i+1}")
            #                             else:
            #                                 st.image(doc.metadata["image"], caption=f"Изображение {i+1}")
            #                         elif isinstance(doc.metadata["image"], bytes):
            #                             st.image(doc.metadata["image"], caption=f"Изображение {i+1}")
            #                     except Exception as e:
            #                         st.error(f"Ошибка загрузки изображения: {str(e)}")

        st.session_state.messages.append({"role": "assistant", "content": answer})