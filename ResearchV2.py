import asyncio
import aiohttp
from bs4 import BeautifulSoup
import openai
import logging
import urllib3
import random
import tiktoken
from datetime import datetime
from prompts import PROMPT_SOFTWARE, PROMPT_ENTERPRISES
from config import API_KEY2

logging.basicConfig(filename="processing.log", level=logging.INFO, format="%(asctime)s - %(message)s")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

openai.api_key = API_KEY2
counter = 0
lock = asyncio.Lock()
encoder = tiktoken.encoding_for_model("gpt-4o-mini")

def count_tokens(text):
    return len(encoder.encode(text))

def clean_text(text):
    import re
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^A-Za-zА-Яа-я0-9 .,!?-]', '', text)
    return text.strip()


def clean_text(text):
    import re
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^A-Za-zА-Яа-я0-9 .,!?-]', '', text)
    return text.strip()


async def fetch_website_content(session, url, retries=3, delay=2):
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
        "Mozilla/5.0 (X11; Linux x86_64) Chrome/118.0.0.0 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) Mobile/15E148 Safari/537.36"
    ]

    for attempt in range(retries):
        headers = {
            "User-Agent": random.choice(user_agents),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://google.com",
            "DNT": "1",
            "Connection": "keep-alive"
        }

        try:
            await asyncio.sleep(random.uniform(0.5, 2))  # Короткая задержка перед первой попыткой
            async with session.get(url, headers=headers, timeout=10, ssl=False) as response:
                if response.status == 403:
                    logging.warning(f"403 Forbidden для {url}, пробуем другой User-Agent (попытка {attempt + 1})")
                    await asyncio.sleep(random.uniform(3, 7))
                    continue

                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                html = await response.text()

                if not html.strip():
                    logging.warning(f"Пустая HTML-страница для {url}")
                    return None

                if "text/html" in content_type:
                    return parse_html(html)
                elif "application/xml" in content_type or "text/xml" in content_type:
                    return parse_html(html, "xml")
                else:
                    logging.warning(f"Неподдерживаемый тип контента ({content_type}) для {url}, пропускаем.")
                    return None

        except Exception as e:
            logging.error(f"Ошибка получения {url} (попытка {attempt + 1}): {e}")
            await asyncio.sleep(random.uniform(2, 5))

    return None


def parse_html(html, parser_type="lxml"):
    soup = BeautifulSoup(html, parser_type)
    key_sections = []

    if soup.title:
        key_sections.append(clean_text(soup.title.get_text(strip=True)))

    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc and meta_desc.get('content'):
        key_sections.append(clean_text(meta_desc.get('content')))

    for tag in ["h1", "h2", "h3", "p", "ul", "ol"]:
        key_sections += [clean_text(el.get_text(strip=True)) for el in soup.find_all(tag)]

    nav = soup.find('nav')
    if nav:
        key_sections += [clean_text(link.get_text(strip=True)) for link in nav.find_all('a') if
                         link.get_text(strip=True)]

    footer = soup.find('footer')
    if footer:
        key_sections.append(clean_text(footer.get_text(strip=True)))

    content = ' '.join(key_sections)

    if len(content) < 100:
        logging.info(f"Недостаточно контента на сайте, длина текста: {len(content)} символов")
        return None

    return content[:15000]


async def analyze_website_content_with_gpt_4o(content, prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI that classifies companies based on their business model."},
                {"role": "user", "content": prompt.format(content=content)}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error(f"Ошибка в GPT-4o-mini: {e}")
        return "-"

async def process_website(session, url, output_file, total_urls, prompt, lock):
    global counter
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"https://{url}"
    domain = url.replace("https://", "").replace("http://", "").split('/')[0]

    content = await fetch_website_content(session, url)
    if content:
        result = await analyze_website_content_with_gpt_4o(content, prompt)
        logging.info(f"{domain}: {result}")  # сохраняет результат анализа в лог
        if result.startswith("+ Relevant"):
            async with lock:
                with open(output_file, 'a', encoding='utf-8') as outfile:
                    outfile.write(f"{domain}\n")

    async with lock:
        counter += 1
        print(f"Прогресс: {counter}/{total_urls} сайтов обработано ({(counter / total_urls) * 100:.2f}%)")


async def process_websites(input_file, output_file, prompt, max_concurrent=50, batch_size=100):
    global counter
    counter = 0
    lock = asyncio.Lock()

    urls = [line.strip() for line in open(input_file, 'r', encoding='utf-8') if line.strip()]
    total_urls = len(urls)

    print(f"Начало обработки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    connector = aiohttp.TCPConnector(limit=max_concurrent, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        # обработка чанками
        for i in range(0, total_urls, batch_size):
            batch = urls[i:i+batch_size]
            await asyncio.sleep(random.uniform(2, 5))  # Задержка перед запуском следующего батча
            tasks = [
                process_website(session, url, output_file, total_urls, prompt, lock)
                for url in batch
            ]
            await asyncio.gather(*tasks)
            print(f"✅ Завершён пакет сайтов с {i+1} по {min(i+batch_size, total_urls)}")

    print(f"Обработка завершена: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Итого обработано: {counter} из {total_urls} ({(counter / total_urls) * 100:.2f}%)")
    print("100% ✔️ Все сайты успешно обработаны!")



input_file = "web.txt"
output_file = "results.txt"
asyncio.run(process_websites(input_file, output_file, PROMPT_SOFTWARE))
