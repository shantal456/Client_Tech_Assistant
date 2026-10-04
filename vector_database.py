from typing import List, Any, Dict
import torch

# Qdrant
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance

# LlamaIndex Core
from llama_index.core import StorageContext, VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.schema import BaseNode

# LlamaIndex Plugins
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.postprocessor import LLMRerank

from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

# Конфигурация (исправлено под vLLM)
from config import (
    QDRANT_URL, EMBEDDING_MODEL, COLLECTION_NAME, 
    TOP_K, TOP_N, QA_TEMPLATE, VLLM_MODEL_NAME, VLLM_URL
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_qdrant_client(url: str = QDRANT_URL) -> QdrantClient:
    return QdrantClient(url=url)


def get_existing_index_from_qdrant(qdrant_client: QdrantClient, collection_name: str) -> VectorStoreIndex:
    """Инициализирует VectorStoreIndex из уже существующей коллекции Qdrant с поддержкой асинхронности."""
    
    async_qdrant_client = AsyncQdrantClient(url=QDRANT_URL)
    
    vector_store = QdrantVectorStore(
        client=qdrant_client, 
        aclient=async_qdrant_client,  # <-- Передаем асинхронный клиент сюда
        collection_name=collection_name, 
        prefer_grpc=False
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

def ensure_qdrant_collection(
    client: QdrantClient, collection_name: str, vector_size: int, distance: Distance = Distance.COSINE
) -> None:
    """Создаёт коллекцию в Qdrant, если её не существует."""
    try:
        client.get_collection(collection_name)
        return
    except Exception:
        pass

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        print(f"✅ Создана коллекция Qdrant '{collection_name}' (size={vector_size}).")
    except Exception as exc:
        print(f"⚠️ Ошибка создания коллекции: {exc}. База попробует создать её автоматически.")


def build_index_to_qdrant(
    nodes: List[BaseNode],
    qdrant_client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    embedding_model_name: str = EMBEDDING_MODEL,
    device: str = DEVICE,
) -> VectorStoreIndex:
    """
    Принимает готовые ноды (чанки) из loader.py, вычисляет эмбеддинги
    и сохраняет их в Qdrant БЕЗ повторного чанкинга.
    """
    if not nodes:
        raise ValueError("Список нод (nodes) пуст.")

    # 1. Инициализация локального эмбеддера
    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name, device=device, normalize=True)
    Settings.embed_model = embed_model

    # 2. Настройка глобального LLM под vLLM
    Settings.llm = OpenAILike(
        model=VLLM_MODEL_NAME,
        api_base=VLLM_URL,
        api_key="fake-key",
        is_chat_model=True,
        temperature=0.1,
        max_tokens=1024
    )

    # 3. Настройка метаданных для каждой ноды
    sample_texts = []
    for i, node in enumerate(nodes):
        node.metadata_template = "{key}: {value}"
        node.metadata_separator = ", "
        # Исключаем header из отправки в LLM, оставляем для поиска
        node.excluded_llm_metadata_keys = ["header"]
        
        if len(sample_texts) < 4:
            sample_texts.append(node.get_content(metadata_mode="none"))

    # 4. Определение размерности вектора
    try:
        sample_embeddings = embed_model.get_text_embedding_batch(sample_texts)
        vector_size = len(sample_embeddings[0])
    except Exception as exc:
        raise RuntimeError(f"Ошибка вычисления тестового эмбеддинга: {exc}")

    # 5. Проверка коллекции
    ensure_qdrant_collection(qdrant_client, collection_name, vector_size, Distance.COSINE)

    # 6. Создание хранилища и запись готовых NODES
    async_qdrant_client = AsyncQdrantClient(url=QDRANT_URL)

    vector_store = QdrantVectorStore(
        client=qdrant_client, 
        aclient=async_qdrant_client,
        collection_name=collection_name, 
        prefer_grpc=False
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"⏳ Запись {len(nodes)} чанков в Qdrant...")
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
    print("✅ Индексация успешно завершена!")
    return index


def make_retriever(index: VectorStoreIndex, top_k: int = TOP_K) -> RetrieverQueryEngine:
    """Собирает QueryEngine с поддержкой семантического поиска Qdrant и родного SBERT BGE Reranker."""
    
    # Настраиваем компактный синтезатор ответов
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT
    )    
    
    # Инициализируем нативный LlamaIndex реранкер на базе sentence-transformers
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base",
        top_n=TOP_N,
        device=DEVICE
    )
    
    # Из Qdrant достаем в 3 раза больше документов (top_k * 3) для последующего реранкинга
    retriever = index.as_retriever(similarity_top_k=top_k * 3)
    
    # Собираем query_engine с официальным нод-постпроцессором LlamaIndex
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[reranker]
    )
    
    # Применяем системный промпт техподдержки
    query_engine.update_prompts({"response_synthesizer:text_qa_template": PromptTemplate(QA_TEMPLATE)})
    
    return query_engine


def print_retrieved_nodes(nodes: List[Any]) -> None:
    """Вспомогательная функция для отладки выданных документов."""
    for i, node in enumerate(nodes):
        source_node = getattr(node, "node", node)
        text = source_node.get_content(metadata_mode="none").replace("\n", " ")
        meta: Dict[str, Any] = source_node.metadata or {}
        score = getattr(node, "score", None)
        score_text = f"{score:.4f}" if isinstance(score, float) else "n/a"
        print(f"🔹 Контекст {i} | score={score_text} | source={meta.get('source')}")
        print(f"    preview={text[:200]}...\n")
