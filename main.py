from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import QueryBundle, Settings

from loader import (
     load_documents_llama_index_and_enrich_metadata,
     chunk_documents
)

from vector_database import (
     create_qdrant_client,
     get_existing_index_from_qdrant,
     build_index_to_qdrant,
     make_retriever
)
from config import DATA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, QDRANT_URL, EMBEDDING_MODEL,COLLECTION_NAME,TOP_K,NODE_TEMPLATE, QA_TEMPLATE, OLLAMA_MODEL_NAME, OLLAMA_URL, LLM_REQUEST_TIMEOUT

DEVICE = "cuda" if (lambda: __import__("torch").cuda.is_available())() else "cpu"




if __name__ == "__main__":
    # Инициализируем клиент Qdrant в самом начале для проверки
    qdrant_client = create_qdrant_client(url=QDRANT_URL)
    

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, device=DEVICE, normalize=True)
    Settings.llm = Ollama(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_URL, request_timeout=LLM_REQUEST_TIMEOUT, temperature=0.1)

    # Существует ли уже коллекция в Qdrant?
    if qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"[База данных] Коллекция '{COLLECTION_NAME}' найдена. Пропускаем загрузку и чанкинг.")
        print("Шаг 3: Подключение к существующей коллекции Qdrant")
        index = get_existing_index_from_qdrant(qdrant_client, collection_name=COLLECTION_NAME)
        print("Индекс успешно восстановлен из Qdrant.")
    else:
        print(f"[База данных] Коллекция '{COLLECTION_NAME}' не найдена. Начинаем полную сборку базы.")
        
        print("Шаг 1: Загрузка, очистка и парсинг документов")
        initial_documents = load_documents_llama_index_and_enrich_metadata(DATA_DIR)

        print("\nШаг 2: Разбивка документов на чанки")
        final_chunks = chunk_documents(initial_documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"Создано чанков: {len(final_chunks)}")

        print("\nПримеры чанков:")
        for i, chunk in enumerate(final_chunks[:5]):
            print(f"\nЧанк {i+1}:")
            content_preview = getattr(chunk, "page_content", "")[:200].replace("\n", " ")
            print(f"  preview: {content_preview}...")
            print(f"  metadata: {getattr(chunk, 'metadata', {})}")

        print("\nШаг 3: Подключение к Qdrant и индексация")
        index = build_index_to_qdrant(final_chunks, qdrant_client, collection_name=COLLECTION_NAME, embedding_model_name=EMBEDDING_MODEL, device=DEVICE)
        print("Индексация завершена. Данные сохранены в Qdrant.")

    
    print("\nШаг 4: Создаём retriever (top-K + FlashRank Rerank)")
    query_engine = make_retriever(index, top_k=TOP_K)

    print("\n=======================================================")
    print("Корпоративный ассистент готов к работе!")
    print("Введите ваш вопрос ниже. Для выхода из программы введите 'выход' или 'exit'.")
    print("=======================================================")

    while True:
        query = input("\nВаш вопрос: ").strip()
        
        if query.lower() in ["выход", "exit", "quit", "q"]:
            print("Завершение работы ассистента. До свидания!")
            break

        if not query:
            print("Вопрос не может быть пустым. Пожалуйста, попробуйте еще раз.")
            continue

        print(f"\nОбработка запроса: '{query}'...")
        
        # 1. Обертываем строку запроса в системный QueryBundle для реранкера
        query_bundle = QueryBundle(query_str=query)
        
        try:
            # 2. Вызываем цепочку ретрива и реранкинга
            initial_nodes = query_engine.retriever.retrieve(query)
            reranked_nodes = query_engine._node_postprocessors[0].postprocess_nodes(
                initial_nodes, query_bundle=query_bundle
            )
            
            # 3. Отправляем запрос в LLM и получаем ответ
            print("Генерация ответа через LLM...")
            response = query_engine.query(query)
            
            # 4. Выводим ответ модели
            print("\n=== ОТВЕТ МОДЕЛИ ===")
            print(response)
            print("=====================")
            
            # 5. Выводим отсортированные и переранжированные источники
            print("\n=== ИСТОЧНИКИ ДЛЯ ЭТОГО ОТВЕТА ===")
            for i, source_node in enumerate(reranked_nodes, start=1):
                score = source_node.score 
                metadata = source_node.node.metadata
                source_file = metadata.get('source', 'Неизвестный файл')
                page = metadata.get('page')
                page_str = f", стр. {page}" if page is not None else ""
                
                print(f"Чанк {i} [Score: {score:.4f}] (Источник: {source_file}{page_str}):")
                print(f"Текст: {source_node.node.get_content()[:150].strip()}...")
            print("==================================")
            
        except Exception as e:
            print(f"\nПроизошла ошибка при обработке запроса: {e}")
            print("Пожалуйста, попробуйте переформулировать вопрос или перезапустите Ollama.")
    