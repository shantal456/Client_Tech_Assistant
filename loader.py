import os
import re
from typing import List
import fitz
from bs4 import BeautifulSoup

from config import DATA_DIR, CHUNK_OVERLAP, CHUNK_SIZE

# Используем только структуры LlamaIndex
from llama_index.core.schema import Document as LlamaIndexDocument
from llama_index.core.node_parser import SentenceSplitter


def clean_html_content(html_text: str) -> str:
    """Удаляет HTML-теги, скрипты и стили только для HTML-документов."""
    try:
        soup = BeautifulSoup(html_text, 'html.parser')
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        return soup.get_text(separator=' ', strip=True)
    except Exception:
        return html_text


def clean_text_spacing(text: str) -> str:
    """Очищает текст от мусорных символов и нормализует пробелы."""
    # Удаление последовательностей нетекстовых символов
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s.,!?;:\'"\-\(\)\[\]{}\/\\]{5,}', ' ', text)
    # Удаление неразрывных пробелов
    text = re.sub(r'[\ufeff\u200b\xa0]+', ' ', text)
    # Нормализация пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    # Удаление одиночных изолированных знаков пунктуации
    text = re.sub(r'(?<=\s)[^a-zA-Zа-яА-Я0-9\s](?=\s)', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def load_and_clean_documents(data_dir: str) -> List[LlamaIndexDocument]:
    """Загрузка документов в формате LlamaIndex с сохранением метаданных."""
    print(f"\nНачинаем загрузку и обработку документов из директории: {data_dir}")
    llama_documents: List[LlamaIndexDocument] = []

    if not os.path.exists(data_dir):
        print(f"❌ Директория {data_dir} не найдена!")
        return llama_documents

    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if not os.path.isfile(file_path):
            continue

        # --- ОБРАБОТКА PDF ---
        if file_name.endswith('.pdf'):
            try:
                doc = fitz.open(file_path)
                for page_num, page in enumerate(doc, start=1):
                    text = page.get_text()
                    cleaned_text = clean_text_spacing(text)
                    
                    if not cleaned_text:
                        continue
                        
                    llama_documents.append(
                        LlamaIndexDocument(
                            text=cleaned_text,
                            metadata={
                                "source": file_name,
                                "page": page_num,
                                "header": os.path.splitext(file_name)[0]
                            }
                        )
                    )
            except Exception as e:
                print(f"Ошибка чтения PDF {file_name}: {e}")

        # --- ОБРАБОТКА HTML ---
        elif file_name.endswith('.html') or file_name.endswith('.htm'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    text_from_html = clean_html_content(html_content)
                    cleaned_text = clean_text_spacing(text_from_html)
                    
                    if cleaned_text:
                        llama_documents.append(
                            LlamaIndexDocument(
                                text=cleaned_text,
                                metadata={
                                    "source": file_name,
                                    "page": 1,
                                    "header": os.path.splitext(file_name)[0]
                                }
                            )
                        )
            except Exception as e:
                print(f"Ошибка чтения HTML {file_name}: {e}")

        # --- ДОБАВЛЕНО: ОБРАБОТКА TXT ---
        elif file_name.endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                    cleaned_text = clean_text_spacing(text_content)
                    
                    if cleaned_text:
                        llama_documents.append(
                            LlamaIndexDocument(
                                text=cleaned_text,
                                metadata={
                                    "source": file_name,
                                    "page": 1,  # У текстового файла одна страница
                                    "header": os.path.splitext(file_name)[0]
                                }
                            )
                        )
            except Exception as e:
                print(f"Ошибка чтения TXT {file_name}: {e}")

    print(f"\nЗагружено частей документов: {len(llama_documents)}.")
    return llama_documents


def chunk_llama_documents(
    documents: List[LlamaIndexDocument], 
    chunk_size: int = CHUNK_SIZE, 
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[LlamaIndexDocument]:
    """Разбивает документы LlamaIndex на чанки с помощью SentenceSplitter."""
    # SentenceSplitter в LlamaIndex умный: он делит по абзацам, предложениям и словам,
    # не разрывая смысловые куски и автоматически наследуя метаданные.
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # get_nodes_from_documents возвращает BaseNode (наследники Document) с метаданными
    nodes = splitter.get_nodes_from_documents(documents)
    
    # Приводим к типу LlamaIndexDocument для единообразия (BaseNode полностью совместим)
    return nodes


if __name__ == "__main__":
    # Для тестов, если config.py еще не обновлен
    # DATA_DIR = "data" ; CHUNK_SIZE = 500 ; CHUNK_OVERLAP = 50

    print("--- Шаг 1: Загрузка и очистка документов ---")
    initial_docs = load_and_clean_documents(DATA_DIR)
    
    print("\n--- Шаг 2: Чанкинг документов через LlamaIndex ---")
    final_chunks = chunk_llama_documents(initial_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"Создано {len(final_chunks)} финальных текстовых чанков.")

    print("\n--- Проверка: Примеры чанков с метаданными ---")
    for i, chunk in enumerate(final_chunks[:3]):
        print(f"\n--- Чанк {i + 1} ---")
        # В LlamaIndex текст хранится в атрибуте .text (вместо .page_content в LangChain)
        content_preview = chunk.text[:150].replace("\n", " ")
        print(f"Контент: {content_preview}...")
        print(f"Метаданные: {chunk.metadata}")
        
