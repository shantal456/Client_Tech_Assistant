import os
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, QueryBundle, set_global_handler


# Подключаем исправленные функции из нашего loader.py
from loader import (
     load_and_clean_documents,
     chunk_llama_documents
)

# Подключаем исправленные функции из нашего vector_database.py
from vector_database import (
     create_qdrant_client,
     get_existing_index_from_qdrant,
     build_index_to_qdrant,
     make_retriever
)

# Конфигурация под vLLM и Qdrant
from config import (
    DATA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, QDRANT_URL, 
    EMBEDDING_MODEL, COLLECTION_NAME, TOP_K, 
    VLLM_MODEL_NAME, VLLM_URL
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # Инициализируем клиент Qdrant в самом начале для проверки
    qdrant_client = create_qdrant_client(url=QDRANT_URL)
    
    # Глобальные настройки LlamaIndex под наш стек (vLLM вместо Ollama)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, device=DEVICE, normalize=True)
    Settings.llm = OpenAILike(
        model=VLLM_MODEL_NAME, 
        api_base=VLLM_URL, 
        api_key="fake-key",
        is_chat_model=True,
        temperature=0.1,
        max_tokens=384
    )

    print("⏳ Подключение к системе мониторинга Langfuse...")
    set_global_handler("langfuse")
    print("✅ Мониторинг Langfuse успешно активирован!")

    # Проверяем, существует ли уже проиндексированная коллекция в Qdrant
    if qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"\n[База данных] Коллекция '{COLLECTION_NAME}' найдена. Пропускаем загрузку и чанкинг.")
        print("👉 Шаг 3: Подключение к существующей коллекции Qdrant")
        index = get_existing_index_from_qdrant(qdrant_client, collection_name=COLLECTION_NAME)
        print("✅ Индекс успешно восстановлен из хранилища.")
    else:
        print(f"\n[База данных] Коллекция '{COLLECTION_NAME}' не найдена. Начинаем полную сборку базы.")
        
        print("👉 Шаг 1: Загрузка, очистка и парсинг документов")
        initial_documents = load_and_clean_documents(DATA_DIR)

        print("\n👉 Шаг 2: Разбивка документов на чанки (LlamaIndex)")
        final_chunks = chunk_llama_documents(initial_documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"Создано чанков: {len(final_chunks)}")

        print("\n🔎 Примеры чанков для верификации:")
        for i, chunk in enumerate(final_chunks[:3]):
            print(f"\nЧанк {i+1}:")
            # Исправлено: используем .text вместо .page_content
            content_preview = chunk.text[:150].replace("\n", " ")
            print(f"  preview: {content_preview}...")
            print(f"  metadata: {chunk.metadata}")

        print("\n👉 Шаг 3: Подключение к Qdrant и векторизация")
        index = build_index_to_qdrant(
            nodes=final_chunks, 
            qdrant_client=qdrant_client, 
            collection_name=COLLECTION_NAME, 
            embedding_model_name=EMBEDDING_MODEL, 
            device=DEVICE
        )
        print("✅ Индексация завершена. Данные успешно сохранены в Qdrant.")

    print("\n👉 Шаг 4: Создаём retriever (Гибридный поиск + Rerank)")
    query_engine = make_retriever(index, top_k=TOP_K)

    print("\n=======================================================")
    print("🚀 Корпоративный RAG-ассистент готов к работе!")
    print("Введите ваш вопрос ниже. Для выхода введите 'exit'.")
    print("=======================================================")

    while True:
        query = input("\n👤 Ваш вопрос: ").strip()
        
        if query.lower() in ["выход", "exit", "quit", "q"]:
            print("Завершение работы ассистента. До свидания!")
            break

        if not query:
            print("Вопрос не может быть пустым. Пожалуйста, попробуйте еще раз.")
            continue

        print(f"⏳ Обработка запроса и генерация ответа...")
        
        try:
            response = query_engine.query(query)
            
            # Выводим ответ модели
            print("\n=== 🤖 ОТВЕТ МОДЕЛИ ===")
            print(response)
            print("=====================")
            
            # Выводим источники, которые отфильтровал и отранжировал реранкер
            print("\n=== 📋 ИСТОЧНИКИ ДЛЯ ЭТОГО ОТВЕТА ===")
            for i, source_node in enumerate(response.source_nodes, start=1):
                score = source_node.score if source_node.score is not None else 0.0
                metadata = source_node.node.metadata
                source_file = metadata.get('source', 'Неизвестный файл')
                page = metadata.get('page')
                page_str = f", стр. {page}" if page is not None else ""
                
                print(f"📍 Источник {i} [Score: {score:.4f}] ({source_file}{page_str}):")
                print(f"   Текст: {source_node.node.get_content()[:120].strip()}...\n")
            print("==================================")
            
        except Exception as e:
            print(f"\n❌ Произошла ошибка при обработке запроса: {e}")
            print("Пожалуйста, проверьте логи контейнеров Qdrant и vLLM.")
