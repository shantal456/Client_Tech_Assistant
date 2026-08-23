import os
import re
from typing import List
from config import DATA_DIR, CHUNK_OVERLAP, CHUNK_SIZE
from bs4 import BeautifulSoup
import fitz

# LlamaIndex imports
from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document as LlamaIndexDocument
from llama_index.readers.file import UnstructuredReader

# Langchain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

   
def clean_document_content(text: str) -> str:
    """
    Очищает текст документа от специфических артефактов MHTML (quoted-printable, HTML/CSS/JS),
    метаданных PDF, бинарных данных, случайных символов и избыточных пробелов.
    """
    # 1. Удаление HTML/CSS/JavaScript с помощью BeautifulSoup
    try:
        soup = BeautifulSoup(text, 'html.parser')
        # Удаляем теги <script> и <style>
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        # Извлекаем чистый текст, удаляя лишние пробелы, создаваемые BeautifulSoup
        text = soup.get_text(separator=' ', strip=True)
    except Exception:
        # Если парсинг HTML не удался, продолжаем с текстом как есть
        pass
       
    # 2. Удаление последовательностей нетекстовых символов и общей "мусорной" пунктуации
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s.,!?;:\'"\-\(\)\[\]{}\/\\]{5,}', ' ', text)
    # Удаление неразрывных пробелов и других юникодных пробелов
    text = re.sub(r'[\ufeff\u200b\xa0]+', ' ', text)

    # 3. Нормализация пробелов и удаление изолированных символов
    text = re.sub(r'\s+', ' ', text).strip() # Заменяем множественные пробелы на один и обрезаем по краям
    # Удаление одиночных неалфавитно-цифровых символов, окруженных пробелами (например, " - ", " $ ")
    text = re.sub(r'(?<=\s)[^a-zA-Zа-яА-Я0-9\s](?=\s)', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip() # Финальная нормализация пробелов

    return text


def load_documents_llama_index_and_enrich_metadata(data_dir: str) -> List[LangchainDocument]:
    """
    Загрузка документов. Извлекает текст постранично 
    с сохранением номеров страниц и имен файлов.
    """
    print(f"\nНачинаем загрузку и обработку документов из директории: {data_dir}")
    all_langchain_documents: List[LangchainDocument] = []

    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if not os.path.isfile(file_path):
            continue

        # Обработка PDF через PyMuPDF
        if file_name.endswith('.pdf'):
            try:
                doc = fitz.open(file_path)
                for page_num, page in enumerate(doc, start=1):
                    text = page.get_text()
                    cleaned_text = clean_document_content(text)
                    
                    if not cleaned_text:
                        continue
                        
                    all_langchain_documents.append(
                        LangchainDocument(
                            page_content=cleaned_text,
                            metadata={
                                "source": file_name,
                                "page": page_num,
                                "header": os.path.splitext(file_name)[0]
                            }
                        )
                    )
            except Exception as e:
                print(f"Ошибка чтения PDF {file_name}: {e}")

        # Обработка HTML (страница всегда None)
        elif file_name.endswith('.html') or file_name.endswith('.htm'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    cleaned_text = clean_document_content(text)
                    if cleaned_text:
                        all_langchain_documents.append(
                            LangchainDocument(
                                page_content=cleaned_text,
                                metadata={
                                    "source": file_name,
                                    "page": None,
                                    "header": os.path.splitext(file_name)[0]
                                }
                            )
                        )
            except Exception as e:
                print(f"Ошибка чтения HTML {file_name}: {e}")

    print(f"\nЗагружено частей: {len(all_langchain_documents)}. Они готовы к чанкингу.")
    return all_langchain_documents

# разбиваем текст на чанки с сохранением метаданных
def chunk_documents(
    documents: List[LangchainDocument], 
    chunk_size: int = CHUNK_SIZE, 
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[LangchainDocument]:
    """
    Разбивает список Langchain Document на более мелкие чанки,
    автоматически передавая оригинальные метаданные каждому новому чанку.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n(?=\s*\d+(?:\.\d+)*(?:[.)])?\s+)", #подпункты
            "\n\n",                   # Абзацы между частями
            ".\s{2,}",                # После предложений
            "\n",                     # Переносы строк
            " ",                      # Пробел
            ""                        # Символы
        ],
        is_separator_regex=True,
        keep_separator=True
    )
    
    chunks = text_splitter.split_documents(documents)
    
    return chunks

# --- Основное выполнение скрипта ---
if __name__ == "__main__":

    # Шаг 1: Загружаем и парсим документы с LlamaIndex, очищаем и обогащаем метаданные
    print("Загрузка, очистка и парсинг документов")
    initial_documents = load_documents_llama_index_and_enrich_metadata(DATA_DIR)
    
    # Шаг 2: Разбиваем извлеченные документы на чанки с Langchain
    print("\nРазбиение очищенных документов на чанки с использованием Langchain RecursiveCharacterTextSplitter...")
    final_chunks = chunk_documents(initial_documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"Создано {len(final_chunks)} финальных текстовых чанков.")

    # --- Проверка: выводим первые несколько чанков для верификации ---
    print("\n--- Проверка: Примеры чанков с метаданными ---")

    for i, chunk in enumerate(final_chunks[:7]):
        print(f"\n--- Чанк {i + 1} ---")

        content_preview = chunk.page_content[:200].replace("\n", " ")
        print(f"Контент (первые 200 символов): {content_preview}...")
        print(f"Метаданные: {chunk.metadata}")