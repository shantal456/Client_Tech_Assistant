import os
import asyncio
import logging
from dotenv import load_dotenv
import langfuse

from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart

# Импортируем LlamaIndex компоненты для инициализации окружения
from llama_index.core import Settings, set_global_handler
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

from vector_database import create_qdrant_client, get_existing_index_from_qdrant, make_retriever
from config import QDRANT_URL, EMBEDDING_MODEL, COLLECTION_NAME, TOP_K, VLLM_MODEL_NAME, VLLM_URL

# Настройка логирования в консоль
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Глобальные переменные для бота и RAG
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
dp = Dispatcher()
query_engine = None

def init_rag_system():
    """Однократная инициализация RAG-компонентов при старте бота."""
    global query_engine
    logger.info("⏳ Инициализация RAG-пайплайна...")
    
    # Включаем Langfuse логирование
    set_global_handler("langfuse")
    
    # Настраиваем локальные модели
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, normalize=True)
    Settings.llm = OpenAILike(
        model=VLLM_MODEL_NAME, 
        api_base=VLLM_URL, 
        api_key="fake-key",
        is_chat_model=True,
        temperature=0.1,
        max_tokens=384
    )
    
    # Подключаемся к существующей коллекции Qdrant
    qdrant_client = create_qdrant_client(url=QDRANT_URL)
    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        raise RuntimeError(f"Коллекция {COLLECTION_NAME} не найдена! Сначала запустите main.py для индексации.")
        
    index = get_existing_index_from_qdrant(qdrant_client, collection_name=COLLECTION_NAME)
    query_engine = make_retriever(index, top_k=TOP_K)
    logger.info("✅ RAG-система успешно запущена и готова к работе.")

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    """Хэндлер команды /start."""
    await message.answer(
        "👋 Здравствуйте! Я умный ассистент техподдержки Банка «Первомайский».\n"
        "Задайте мне любой вопрос по тарифам, картам, кредитам или услугам банка, и я найду ответ в базе знаний."
    )

@dp.message()
async def handle_user_query(message: types.Message):
    user_text = message.text.strip()
    if not user_text:
        return

    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
    logger.info(f"👤 Запрос от пользователя {message.from_user.id}: {user_text}")

    try:
        # Асинхронная генерация ответа через RAG
        response = await query_engine.aquery(user_text)
        
        try:
            langfuse.flush()
        except Exception as lf_err:
            logger.warning(f"Не удалось выполнить flush в Langfuse: {lf_err}")
        # ==================================================================

        # Формируем текст ответа
        reply_text = f"{response}\n\n"
        
        # Выводим только реально подходящие файлы
        if response.source_nodes:
            sources = set()
            for node in response.source_nodes:
                # Если скор ноды после реранкера высокий, добавляем в источники
                if node.score and node.score > 0.3: 
                    sources.add(node.node.metadata.get('source', 'База знаний'))
            
            if sources:
                reply_text += f"Использованные источники: {', '.join(sources)}"


        # Отправляем ответ пользователю
        await message.answer(reply_text)
        
    except Exception as e:
        logger.error(f"❌ Ошибка при обработке запроса: {e}")
        await message.answer("⚠️ К сожалению, произошла техническая ошибка при обработке запроса.")


async def main():
    # 1. Инициализируем RAG
    init_rag_system()
    
    # 2. Инициализируем бота
    bot = Bot(token=BOT_TOKEN)
    
    # 3. Запускаем long polling
    logger.info("🚀 Запуск Telegram-бота...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен.")
