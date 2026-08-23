from typing import List, Any, Dict

# Qdrant
from qdrant_client import QdrantClient
try:
    from qdrant_client.models import VectorParams, Distance
except Exception:
    # fallback for older/newer package layouts
    from qdrant_client.http.models import VectorParams, Distance  # type: ignore

# LlamaIndex
from llama_index.core import StorageContext, VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Document as LlamaDocument
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.flashrank_rerank import FlashRankRerank

from loader import (
     load_documents_llama_index_and_enrich_metadata,
     chunk_documents
)
from config import QDRANT_URL, EMBEDDING_MODEL,COLLECTION_NAME,TOP_K,TOP_N, QA_TEMPLATE, OLLAMA_MODEL_NAME, OLLAMA_URL, LLM_REQUEST_TIMEOUT

DEVICE = "cuda" if (lambda: __import__("torch").cuda.is_available())() else "cpu"

def create_qdrant_client(url: str = QDRANT_URL) -> QdrantClient:
    return QdrantClient(url=url)

def get_existing_index_from_qdrant(qdrant_client: QdrantClient, collection_name: str) -> VectorStoreIndex:
    """
    Инициализирует VectorStoreIndex из уже существующей коллекции Qdrant 
    БЕЗ повторной векторизации и загрузки документов.
    """
    # Создаем векторное хранилище, привязанное к существующей коллекции
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, prefer_grpc=False)
    
    # Передаем его в контекст хранилища
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Восстанавливаем индекс из существующего контекста (документы передавать не нужно)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
    return index

def ensure_qdrant_collection(
    client: QdrantClient, collection_name: str, vector_size: int, distance: Distance = Distance.COSINE
) -> None:
    # Создаёт коллекцию, если она не существует
    try:
        # get_collection вернёт словарь/объект если существует
        client.get_collection(collection_name)
        # существует -> ничего не делаем
        return
    except Exception:
        pass

    # Создаём коллекцию
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        print(f"Создана коллекция Qdrant '{collection_name}' (size={vector_size}, distance={distance}).")
    except TypeError:
        # В некоторых версиях API сигнатура может отличаться
        try:
            client.create_collection(
                collection_name=collection_name,
                vector_size=vector_size,
                distance=distance,
            )
            print(f"Создана коллекция Qdrant '{collection_name}' (size={vector_size}, distance={distance}).")
        except Exception as exc:
            print(f"Не удалось явно создать коллекцию: {exc}. Коллекция может быть создана автоматически при вставке.")

def build_index_to_qdrant(
    final_chunks: List[Any],
    qdrant_client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    embedding_model_name: str = EMBEDDING_MODEL,
    device: str = DEVICE,
) -> VectorStoreIndex:
    """
    Индексирует final_chunks в Qdrant через llama_index, сохраняя метаданные.
    final_chunks: список объектов с .page_content (str) и .metadata (dict).
    Возвращает созданный VectorStoreIndex.
    """
    if not final_chunks:
        raise ValueError("final_chunks пустой список")

    # 1 Создаём эмбеддер
    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name, device=device, normalize=True)

    # 2 Конвертируем чанки в LlamaIndex Document'ы
    llama_docs: List[LlamaDocument] = []
    sample_texts: List[str] = []
    for i, chunk in enumerate(final_chunks):
        text = getattr(chunk, "page_content", None)
        if text is None:
            text = str(chunk)
        metadata = getattr(chunk, "metadata", None) or {}
        doc_id = f"{metadata.get('source', 'doc')}_chunk_{i}"

        doc = LlamaDocument(text=text, metadata=metadata, id_=doc_id)
        #Настройка формата метаданных для этого документа
        doc.metadata_template = "{key}: {value}"
        doc.metadata_separator = ", стр. "
        doc.excluded_llm_metadata_keys = ["header"] 
        llama_docs.append(doc)
        if len(sample_texts) < 4:
            sample_texts.append(text)

    # 3 Оценка размера вектора (vector_size) на основе sample_embeddings
    try:
        sample_embeddings = embed_model.get_text_embedding_batch(sample_texts)
    except Exception as exc:
        raise RuntimeError("Ошибка при вычислении эмбеддингов для sample_texts: " + str(exc))
    if not sample_embeddings or not isinstance(sample_embeddings[0], (list, tuple)):
        raise RuntimeError("Не удалось получить корректные эмбеддинги для определения vector_size.")
    vector_size = len(sample_embeddings[0])

    # 4 Создаём/проверяем коллекцию в Qdrant
    ensure_qdrant_collection(qdrant_client, collection_name, vector_size, Distance.COSINE)
    Settings.embed_model = embed_model
    Settings.llm = Ollama(
    model=OLLAMA_MODEL_NAME, 
    base_url=OLLAMA_URL,
    request_timeout=LLM_REQUEST_TIMEOUT,
    temperature=0.1,
    context_window=4096
    )

    # 5 Создаём QdrantVectorStore и StorageContext для llama_index
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, prefer_grpc=False)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 6 Создаём индекс
    index = VectorStoreIndex.from_documents(
        documents=llama_docs, 
        storage_context=storage_context
    )
    return index

def make_retriever(index: VectorStoreIndex, top_k: int = TOP_K) -> RetrieverQueryEngine:
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.SIMPLE_SUMMARIZE
    )    
    # Инициализация FlashRank
    reranker = FlashRankRerank(
        top_n=TOP_N,  # сколько лучших чанков пропустить в LLM после переранжирования
        model="ms-marco-MultiBERT-L-12"
    )
    
    # Qdrant достает top_k * 3 чанков
    retriever = index.as_retriever(similarity_top_k=top_k * 3)
    
    # Собираем query_engine с постобработкой
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[reranker]
    )
    
    query_engine.update_prompts({"response_synthesizer:text_qa_template": PromptTemplate(QA_TEMPLATE)})
    
    return query_engine


def print_retrieved_nodes(nodes: List[Any]) -> None:
    for i, node in enumerate(nodes):
        source_node = getattr(node, "node", node)
        text = source_node.get_content(metadata_mode="none").replace("\n", " ")
        meta: Dict[str, Any] = source_node.metadata or {}
        score = getattr(node, "score", None)
        score_text = f"{score:.4f}" if isinstance(score, float) else "n/a"
        print(f"Контекст {i} score={score_text} id={meta.get('id') or meta.get('source')}")
        print(f"    metadata={meta}")
        print(f"    preview={text[:400]}...\n")