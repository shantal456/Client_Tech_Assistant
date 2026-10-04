import sys
import requests
from config import VLLM_MODEL_NAME, EXPECTED_QDRANT_VERSION


def check_vllm():
    print("⏳ Проверка vLLM...", end=" ")
    try:
        # Проверяем базовую доступность OpenAI-совместимого API vLLM
        # Извне контейнера обращаемся по порту 8080, как указано в docker-compose
        r = requests.get("http://localhost:8080/v1/models", timeout=30)
        if r.status_code == 200:  # noqa: PLR2004
            print("✅ OK")
        else:
            print(f"❌ Ошибка: статус {r.status_code}")
            return False

        # Проверяем, загружена ли именно наша модель
        print(f"⏳ Проверка модели {VLLM_MODEL_NAME} во vLLM...", end=" ")
        models_data = r.json().get("data", [])
        loaded_models = [m["id"] for m in models_data]

        if VLLM_MODEL_NAME in loaded_models:
            print("✅ Модель загружена и готова")
            return True
        
        print(f"❌ Модель не найдена в системе. Загруженные во vLLM: {loaded_models}")
        print("👉 Проверьте логи контейнера vLLM: docker compose logs vllm")
        return False

    except Exception as e:
        print(f"❌ Ошибка соединения (убедитесь, что vLLM запустился и порт 8080 открыт): {e}")
        return False


def check_qdrant():
    print("⏳ Проверка Qdrant...", end=" ")
    try:
        r = requests.get("http://localhost:6333/collections", timeout=30)
        if r.status_code == 200:  # noqa: PLR2004
            print(f"✅ OK (Коллекций: {len(r.json()['result']['collections'])})")
            version_response = requests.get("http://localhost:6333/", timeout=30)
            version = version_response.json().get("version")
            
            # Отрезаем букву 'v' из тега (например, v1.16.2 -> 1.16.2), если она есть в ответе сервера
            clean_version = version.lstrip('v')
            clean_expected = EXPECTED_QDRANT_VERSION.lstrip('v')
            
            if clean_version != clean_expected:
                print(
                    "⚠️ Версия Qdrant отличается от ожидаемой: "
                    f"server={version}, expected={EXPECTED_QDRANT_VERSION}"
                )
                print("👉 Пересоздайте контейнеры при необходимости: docker compose down -v && docker compose up -d")
                return False
            print(f"✅ Версия Qdrant: {version}")
            return True
        print(f"❌ Ошибка: статус {r.status_code}")
        return False
    except Exception as e:
        print(f"❌ Ошибка соединения: {e}")
        return False


if __name__ == "__main__":
    vllm_ok = check_vllm()
    qdrant_ok = check_qdrant()

    if vllm_ok and qdrant_ok:
        print("\n🚀 Все системы готовы к работе!")
    else:
        print("\n⚠️ Есть проблемы с сервисами. Исправьте их перед продолжением.")
        sys.exit(1)
