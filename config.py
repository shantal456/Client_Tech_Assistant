import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR="data"
OUTPUT_DATA_DIR="output_data"

# Параметры чанкования
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


EXPECTED_QDRANT_VERSION = "1.16.2"

QDRANT_URL = "http://localhost:6333"
QDRANT_SERVER_VERSION = "1.16.2"
OLLAMA_URL = "http://localhost:11434"
COLLECTION_NAME = "docs_llamaindex_2"
OLLAMA_MODEL_NAME = "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M"
EMBEDDING_MODEL = "cointegrated/rubert-tiny2"
LLM_REQUEST_TIMEOUT = 600.0
TOP_K=5
TOP_N=3

NODE_TEMPLATE ="""\
Контекст {i} score={score_text} id={meta.get('id') or meta.get('source')}")
Источник: {meta}
{text}
"""

QA_TEMPLATE = """\
Ты — корпоративный ассистент гимназии 33.
Ответь на вопрос пользователя, используя ТОЛЬКО предоставленный ниже контекст.
Если в контексте нет информации, скажи "В документах нет информации об этом".
Не придумывай факты.
"ОБЯЗАТЕЛЬНО: Если в документах есть ответ, подтверждай каждый факт ссылкой на источник. Формат ссылки должен быть строго таким: [Название файла, стр. X] или [Название html-статьи]."

Контекст:
---------------------
{context_str}
---------------------

Вопрос: {query_str}

Ответ:
"""