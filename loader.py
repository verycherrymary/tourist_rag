import pandas as pd
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_split_markdown(filepath='https://drive.google.com/u/0/uc?id=1JQswhvNz6yNKKzJW0nrXU7AmUQaGevxA&export=download'):
    # Загрузка данных
    data_cities = pd.read_csv(filepath)
    
    # Создание документов без прогресс-бара
    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(
            page_content=f"{row['City']} | {row['Name']} | {row['description']}",
            metadata={
                "longitude": row['Lon'],
                "latitude": row['Lat'],
                "image": row['image'],
                # "english_description": row['en_txt']
            }
        )
        for _, row in data_cities.iterrows()  # Убрали tqdm
    ]

    # Настройки разделителя текста
    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    # Инициализация разделителя текста
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    # Разделение документов
    docs_processed = []
    for doc in RAW_KNOWLEDGE_BASE:
        docs_processed += text_splitter.split_documents([doc])

    return docs_processed           