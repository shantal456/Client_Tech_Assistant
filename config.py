import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
OUTPUT_DATA_DIR = "output_data"

# Параметры чанкования
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

raw_vllm_url = os.getenv("VLLM_URL", "http://localhost:8080")
if not raw_vllm_url.endswith("/v1"):
    VLLM_URL = f"{raw_vllm_url.rstrip('/')}/v1"
else:
    VLLM_URL = raw_vllm_url
VLLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# Параметры поиска
TOP_K = 5
TOP_N = 3  # Сколько документов отдавать в LLM после реранкинга

EXPECTED_QDRANT_VERSION = "1.16.2"

# ИСПРАВЛЕНО: Сначала ищет имя контейнера в Docker, если нет — берет localhost хоста
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_SERVER_VERSION = "1.16.2"
COLLECTION_NAME = "docs_llamaindex"
EMBEDDING_MODEL = "cointegrated/rubert-tiny2"
LLM_REQUEST_TIMEOUT = 600.0

NODE_TEMPLATE = """\
Контекст {i} score={score_text} id={meta.get('id') or meta.get('source')}")
Источник: {meta}
{text}
"""

QA_TEMPLATE = (
    "Ты — квалифицированный ассистент технической поддержки Банка «Первомайский».\n"
    "Твоя задача — помочь пользователю, используя ТОЛЬКО предоставленные документы.\n\n"
    "ДОКУМЕНТЫ С БАЗЫ ЗНАНИЙ:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "ИНСТРУКЦИЯ:\n"
    "1. Внимательно изучи документы выше. Если в них есть ответ на вопрос, подробно ответь, опираясь на факты.\n"
    "2. Обязательно укажи в тексте ответа, из какого именно файла (например, doc_03_deposits.txt) взята информация.\n"
    "3. Если в документах вообще нет упоминания темы вопроса, тогда и только тогда ответь: 'К сожалению, в базе знаний нет информации об этом'.\n\n"
    "Вопрос пользователя: {query_str}\n"
    "Ответ: "
)
