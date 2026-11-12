from bs4          import BeautifulSoup
from datetime     import datetime
from pathlib      import Path
from urllib.parse import urlparse

from ddgs         import DDGS
from openai       import AsyncOpenAI
from dotenv       import load_dotenv

import asyncio
import json
import os

import random
import requests

load_dotenv(Path(__file__).parent / ".env")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8",
}

RESULTS_FILE = "results.json"
RESTAURANTS_FILE = "restaurants.json"
MODEL = "google/gemma-4-26b-a4b-it"
MAX_CHARS = 4000

client = AsyncOpenAI(
    base_url="https://polza.ai/api/v1",
    api_key=os.environ.get("POLZA_AI_API_KEY"),
)


# ─────────── 1. ПОИСК ───────────
async def search_web(query: str, limit: int = 10) -> None:
    """Ищет через ddgs (в отдельном потоке, чтобы не блокировать event loop)."""
    try:
        # ddgs синхронный, поэтому выносим его в поток
        raw_results = await asyncio.to_thread(
            lambda: DDGS().text(query, max_results=limit, region="ru-ru")
        )
    except Exception as e:
        print(f"Ошибка поиска ddgs: {e}")
        return

    if not raw_results:
        print("Поиск не дал результатов.")
        return

    # Приводим ключи ddgs к формату вашего пайплайна (title, url, snippet)
    hits = [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in raw_results
    ]

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"query": query, "fetched_at": datetime.now().isoformat(), "results": hits},
            f, ensure_ascii=False, indent=2,
        )

    print(f"Найдено {len(hits)} ссылок, сохранено в {RESULTS_FILE}")


# ─────────── 2. ЗАГРУЗКА URL ───────────
def load_items(filepath: str = RESULTS_FILE) -> tuple[list[str], dict[str, str]]:
    """Возвращает (уникальные URL, {url: title}) из results.json."""
    if not os.path.exists(filepath):
        print(f"Файл не найден: {filepath}")
        return [], {}

    with open(filepath, "r", encoding="utf-8") as f:
        items = json.load(f).get("results", [])

    title_by_url = {item["url"]: item.get("title", "") for item in items if item.get("url")}
    return list(title_by_url), title_by_url


# ─────────── 3. РАЗБОР САЙТОВ ───────────
def fetch_text(url: str) -> str | None:
    """Скачивает страницу и возвращает очищенный текст. При ошибке — None."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


# ─────────── 4. ИИ ───────────
async def extract_restaurant_data(text: str) -> dict | None:
    """Возвращает {phone, email, contacts} или None, если это не ресторан."""
    if not text:
        return None

    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — парсер контактных данных. Определи, является ли текст сайтом ресторана. "
                        "Если да — извлеки телефон, email и любые контакты. "
                        "Если это не ресторан — верни is_restaurant: false. "
                        "Если контактов нет — оставь поля пустыми."
                    ),
                },
                {"role": "user", "content": text[:MAX_CHARS]},
            ],
            temperature=0.0,
            max_tokens=600,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "restaurant_data",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "is_restaurant": {"type": "boolean"},
                            "phone": {"type": "string"},
                            "email": {"type": "string"},
                            "contacts": {"type": "string"},
                        },
                        "required": ["is_restaurant", "phone", "email", "contacts"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        data = json.loads(resp.choices[0].message.content)
        return data if data.get("is_restaurant") else None
    except Exception as e:
        print(f"Ошибка ИИ: {e}")
        return None


# ─────────── 5. СОХРАНЕНИЕ ───────────
def save_restaurant(url: str, info: dict, title: str = "") -> bool:
    """Добавляет ресторан в файл. Возвращает True, если добавлен (не дубль)."""
    store = {"restaurants": [], "updated_at": ""}
    if os.path.exists(RESTAURANTS_FILE):
        try:
            with open(RESTAURANTS_FILE, "r", encoding="utf-8") as f:
                store = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    if any(item["url"] == url for item in store.get("restaurants", [])):
        return False

    store.setdefault("restaurants", []).append({
        "url": url,
        "domain": urlparse(url).netloc,
        "title": title,
        "phone": info.get("phone", ""),
        "email": info.get("email", ""),
        "contacts": info.get("contacts", ""),
        "saved_at": datetime.now().isoformat(),
    })
    store["updated_at"] = datetime.now().isoformat()

    with open(RESTAURANTS_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
    return True


# ─────────── КОМПОЗИЦИЯ ───────────
async def main() -> None:
    if not os.environ.get("POLZA_AI_API_KEY"):
        print("POLZA_AI_API_KEY не задан в .env")
        return

    await search_web("заказать еду в ресторане", limit=100)
    await asyncio.sleep(random.uniform(2.0, 4.0))

    urls, title_by_url = load_items()
    print(f"Загружено {len(urls)} уникальных URL")

    loaded = failed = restaurants = 0

    for i, url in enumerate(urls, 1):
        text = fetch_text(url)
        if not text:
            failed += 1
            await asyncio.sleep(random.uniform(1.5, 3.0))
            continue

        loaded += 1
        info = await extract_restaurant_data(text)

        if info:
            restaurants += 1
            print(f"✅ РЕСТОРАН: {url}")
            print(f"   📞 {info.get('phone') or '—'}  ✉️ {info.get('email') or '—'}")
            save_restaurant(url, info, title=title_by_url.get(url, ""))
        else:
            print(f"❌ не ресторан: {url}")

        await asyncio.sleep(random.uniform(1.5, 3.0))
        if i % 7 == 0 and i < len(urls):
            pause = random.uniform(15.0, 30.0)

            print(f"⏸ Пауза {pause:.0f} сек...")
            await asyncio.sleep(pause)

    print(f"\nИтог: загружено {loaded}, не загружено {failed}, ресторанов — {restaurants}")
    print(f"Сохранено в {RESTAURANTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())