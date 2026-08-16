import os
import re
from typing import List, Dict, Any, Optional
from config import DATA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, OUTPUT_DATA_DIR
from bs4 import BeautifulSoup

# LlamaIndex imports
from llama_index.core import SimpleDirectoryReader
#from llama_index.readers.file import PypdfReader, DocxReader, HTMLReader # Специфичные ридеры для контроля
from llama_index.core.schema import Document as LlamaIndexDocument
from llama_index.readers.file import UnstructuredReader

# Langchain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

# --- Функция для очистки контента документов ---
def clean_document_content(text: str) -> str:
    """
    Очищает текст документа от специфических артефактов MHTML (quoted-printable, HTML/CSS/JS),
    метаданных PDF, бинарных данных, случайных символов и избыточных пробелов.
    """
    # 2. Удаление HTML/CSS/JavaScript (актуально для MHTML и некоторых PDF, содержащих встроенный HTML)
    # Используем BeautifulSoup для более надежного парсинга и удаления тегов
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
       
    # 6. Удаление последовательностей нетекстовых символов и общей "мусорной" пунктуации
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s.,!?;:\'"\-\(\)\[\]{}\/\\]{5,}', ' ', text)
    # Удаление неразрывных пробелов и других юникодных пробелов
    text = re.sub(r'[\ufeff\u200b\xa0]+', ' ', text)

    # 7. Нормализация пробелов и удаление изолированных символов
    text = re.sub(r'\s+', ' ', text).strip() # Заменяем множественные пробелы на один и обрезаем по краям
    # Удаление одиночных неалфавитно-цифровых символов, окруженных пробелами (например, " - ", " $ ")
    text = re.sub(r'(?<=\s)[^a-zA-Zа-яА-Я0-9\s](?=\s)', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip() # Финальная нормализация пробелов

    return text

# --- Улучшенная функция для идентификации заголовка ---
def identify_header_from_text(text: str, default_header: str) -> str:
    """
    Пытается определить заголовок из начала очищенного текста.
    Принимает очищенный текст и заголовок по умолчанию (имя файла).
    """
    # Разделяем текст на строки и берем непустые
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    for line in lines[:5]: # Проверяем первые 5 непустых строк
        if 10 < len(line) < 150:
            # Считаем долю алфавитно-цифровых символов
            alnum_chars = sum(c.isalnum() for c in line)
            if len(line) > 0 and (alnum_chars / len(line)) > 0.7: # Более 70% букв/цифр
                # Дополнительная проверка: не должен быть похож на URL или путь файла,
                # и не должен начинаться с цифры, если это не явно год или номер документа
                if not re.match(r'^(http|www|file|/)', line, re.IGNORECASE) and \
                   not re.match(r'^\d{1,4}[.\-/\\]', line): # Не год/номер документа
                    return line
    
    return default_header


def load_documents_llama_index_and_enrich_metadata(
    data_dir: str,
    save_extracted_texts_to_dir: Optional[str] = None
) -> List[LangchainDocument]:
    """
    Загружает документы из директории, используя UnstructuredFileExtractor для PDF,
    очищает их контент, обогащает метаданные и преобразует в формат Langchain Document.
    """
    print(f"\nНачинаем загрузку и обработку документов из директории: {data_dir}")

    # Инициализация UnstructuredFileExtractor для PDF-файлов.
    unstructured_reader = UnstructuredReader()

    file_extractor = {
    ".pdf": unstructured_reader
    }

    all_llama_documents: List[LlamaIndexDocument] = []
    try:
        # Инициализируем SimpleDirectoryReader с нашим настроенным file_extractor
        loader = SimpleDirectoryReader(
            input_dir=data_dir,
            file_extractor=file_extractor
        )
        all_llama_documents = loader.load_data()
        print(f"Всего собрано {len(all_llama_documents)} исходных документов/частей (LlamaIndex) с помощью SimpleDirectoryReader и Unstructured.")
    except Exception as e:
        print(f"Ошибка при загрузке файлов с SimpleDirectoryReader и Unstructured: {e}")
        print("Пожалуйста, убедитесь, что 'unstructured', 'unstructured-client' и 'llama-index-readers-file' установлены:")
        print("pip install unstructured unstructured-client llama-index-readers-file")
        return []

    all_langchain_documents: List[LangchainDocument] = []
    skipped_documents_count = 0
    saved_texts_count = 0

    if save_extracted_texts_to_dir:
        os.makedirs(save_extracted_texts_to_dir, exist_ok=True)
        print(f"Очищенные тексты будут сохранены в: {os.path.abspath(save_extracted_texts_to_dir)}")

    for i, li_doc in enumerate(all_llama_documents):
        # Unstructured может добавлять свои метаданные, например 'filename', 'page_number'
        file_name = li_doc.metadata.get('filename', li_doc.metadata.get('file_name', f'unknown_file_{i}'))
        page_label = li_doc.metadata.get('page_number', li_doc.metadata.get('page_label', None))
        
        initial_text_len = len(li_doc.text)
        cleaned_text = clean_document_content(li_doc.text)

        output_filename_base = os.path.splitext(file_name)[0]
        if page_label:
            output_filename_base += f"_page_{page_label}"
        output_filename = f"{output_filename_base}.txt"

        if not cleaned_text:
            print(f"  Внимание: Документ '{file_name}' (стр: {page_label or 'N/A'}) был полностью очищен (был {initial_text_len} символов) и пропущен.")
            skipped_documents_count += 1
            continue
        
        default_header = os.path.splitext(file_name)[0]
        header = identify_header_from_text(cleaned_text, default_header)
        
        langchain_doc = LangchainDocument(
            page_content=cleaned_text,
            metadata={
                "source": file_name,
                "page": page_label,
                "header": header
            }
        )
        all_langchain_documents.append(langchain_doc)

        if save_extracted_texts_to_dir:
            output_filepath = os.path.join(save_extracted_texts_to_dir, output_filename)
            try:
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                saved_texts_count += 1
            except IOError as e:
                print(f"  Ошибка при сохранении текста '{file_name}' в '{output_filepath}': {e}")
        
    print(f"\nПосле очистки и обогащения метаданных, {len(all_langchain_documents)} документов готовы к чанкингу.")
    if skipped_documents_count > 0:
        print(f"Всего пропущено {skipped_documents_count} документов/страниц из-за пустого контента после очистки.")
    if save_extracted_texts_to_dir:
        print(f"Всего сохранено {saved_texts_count} очищенных текстовых файлов в {os.path.abspath(save_extracted_texts_to_dir)}.")
        
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
    print("\n--- Запуск Части 1: Загрузка, Парсинг и Чанкинг Документов ---\n")

    # Шаг 1: Загружаем и парсим документы с LlamaIndex, очищаем и обогащаем метаданные
    print("Загрузка, очистка и парсинг документов с использованием LlamaIndex...")
    initial_documents = load_documents_llama_index_and_enrich_metadata(DATA_DIR, save_extracted_texts_to_dir=OUTPUT_DATA_DIR)
    
    # Шаг 2: Разбиваем извлеченные документы на чанки с Langchain
    print("\nРазбиение очищенных документов на чанки с использованием Langchain RecursiveCharacterTextSplitter...")
    final_chunks = chunk_documents(initial_documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"Создано {len(final_chunks)} финальных текстовых чанков.")

    # --- Проверка: Выводим первые несколько чанков для верификации ---
    print("\n--- Проверка: Примеры чанков с метаданными ---")
    for i, chunk in enumerate(final_chunks[:7]): # Выводим первые 7 чанков для разнообразия
        print(f"\n--- Чанк {i+1} ---")
        # Выводим первые 200 символов, заменяя переносы строк для лучшей читаемости в консоли
        print(f"Контент (первые 200 символов): {chunk.page_content[:2000].replace('\n', ' ')}...")
        print(f"Метаданные: {chunk.metadata}")
        if i == 6:
            break