import sys

import requests
from config import OLLAMA_MODEL_NAME, EXPECTED_QDRANT_VERSION




def check_ollama():
    print("⏳ Проверка Ollama...", end=" ")
    try:
        # Проверка доступности API
        r = requests.get("http://localhost:11434/", timeout=30)
        if r.status_code == 200:  # noqa: PLR2004
            print("✅ OK")
        else:
            print(f"❌ Ошибка: статус {r.status_code}")
            return False

        # Проверка наличия локальной Ollama-модели
        print(f"⏳ Проверка модели {OLLAMA_MODEL_NAME}...", end=" ")
        r = requests.get("http://localhost:11434/api/tags", timeout=30)
        models = [m["name"] for m in r.json()["models"]]

        # Ollama может вернуть точное имя или вариант с дополнительным суффиксом.
        # Ищем вхождение строки
        if any(OLLAMA_MODEL_NAME in m for m in models):
            print("✅ Модель найдена")
            return True
        print(f"❌ Модель не найдена. Доступные: {models}")
        print(f"👉 Выполните: ollama pull {OLLAMA_MODEL_NAME}")
        return False

    except Exception as e:
        print(f"❌ Ошибка соединения: {e}")
        return False


def check_qdrant():
    print("⏳ Проверка Qdrant...", end=" ")
    try:
        r = requests.get("http://localhost:6333/collections", timeout=30)
        if r.status_code == 200:  # noqa: PLR2004
            print(f"✅ OK (Коллекций: {len(r.json()['result']['collections'])})")
            version_response = requests.get("http://localhost:6333/", timeout=30)
            version = version_response.json().get("version")
            if version != EXPECTED_QDRANT_VERSION:
                print(
                    "⚠️ Версия Qdrant отличается от qdrant-client: "
                    f"server={version}, expected={EXPECTED_QDRANT_VERSION}"
                )
                print("👉 Пересоздайте контейнеры: docker compose down -v && docker compose up -d")
                return False
            print(f"✅ Версия Qdrant: {version}")
            return True
        print(f"❌ Ошибка: статус {r.status_code}")
        return False
    except Exception as e:
        print(f"❌ Ошибка соединения: {e}")
        return False


if __name__ == "__main__":
    ollama_ok = check_ollama()
    qdrant_ok = check_qdrant()

    if ollama_ok and qdrant_ok:
        print("\n🚀 Все системы готовы к работе!")
    else:
        print("\n⚠️ Есть проблемы с сервисами. Исправьте их перед продолжением.")
        sys.exit(1)