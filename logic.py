import requests
from bs4 import BeautifulSoup
import openai
import time
import logging
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


# Настройка логирования
logging.basicConfig(filename="processing.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Отключение предупреждений о проверке SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Укажите ваш API-ключ
API_KEY = os.getenv("API_KEY")

EXCLUDE_KEYWORDS = ["microsoft 365", "dynamics 365", "office 365"]


def fetch_website_content(url):
    """
    Получает текстовый контент с указанного URL.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, timeout=15, headers=headers, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        text = " ".join(text.split())
        if len(text) < 500:
            print(f"{url}: Недостаточно содержимого для анализа.")
            return None
        return text[:5000]  # Увеличено ограничение до 10,000 символов
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при загрузке сайта {url}: {e}")
        logging.error(f"Ошибка при загрузке сайта {url}: {e}")
        return None


def clean_text(content):
    """
    Убирает ненужные фрагменты текста.
    """
    unwanted_phrases = [
        "terms of service", "privacy policy", "cookies",
        "contact us", "about us", "press releases"
    ]
    for phrase in unwanted_phrases:
        content = content.replace(phrase, "")
    return content


def contains_excluded_keywords(content):
    """
    Проверяет, содержит ли текст ключевые слова, связанные с исключенными продуктами.
    """
    content_lower = content.lower()
    for keyword in EXCLUDE_KEYWORDS:
        if keyword in content_lower:
            return True
    return False


def analyze_website_content_with_gpt(content):
    """
    Анализирует текст с помощью GPT и предоставляет объяснение для релевантных сайтов.
    """
    prompt = f"""
        Analyze the following website text and determine whether the company is a SaaS company.

        ### Output format:
        - If SaaS:  
          "+ SaaS: {{Detailed explanation of why the product is relevant}}."
        - If not Saas:  
          "- Not SaaS: {{Reason for exclusion}}."

        ---
    {content}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:  # Общий блок для всех исключений
        print(f"Ошибка при взаимодействии с API OpenAI: {e}")
        logging.error(f"Ошибка при взаимодействии с API OpenAI: {e}")
        return "-"


def process_website(url, output_file):
    """
    Обрабатывает один сайт: скачивает контент, проверяет релевантность и анализирует.
    """
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"https://{url}"

    domain = url.replace("https://", "").replace("http://", "").split('/')[0]

    print(f"Обработка сайта: {domain}")
    logging.info(f"Обработка сайта: {domain}")

    content = fetch_website_content(url)
    if not content:
        print(f"{domain}: Сайт недоступен или текст не найден.")
        logging.warning(f"{domain}: Сайт недоступен или текст не найден.")
        return None

    content = clean_text(content)

    # Фильтрация по ключевым словам
    if contains_excluded_keywords(content):
        print(f"{domain}: Найдены ключевые слова исключения. Пропущено.")
        logging.info(f"{domain}: Найдены ключевые слова исключения. Пропущено.")
        return None

    result = analyze_website_content_with_gpt(content)
    if result.startswith("+ SaaS"):
        print(f"{domain}: {result}")
        # Запись только домена сайта в файл
        with open(output_file, 'a', encoding='utf-8') as outfile:
            outfile.write(f"{domain}\n")
        return f"{domain}"
    return None


def process_websites(input_file, output_file, max_threads=5):
    """
    Обрабатывает список сайтов с использованием многопоточности.
    """
    urls = []
    # Открытие входного файла с явной кодировкой UTF-8
    with open(input_file, 'r', encoding='utf-8') as infile:
        urls = [line.strip() for line in infile if line.strip()]

    # Используем ThreadPoolExecutor для многопоточности
    with ThreadPoolExecutor(max_threads) as executor:
        future_to_url = {executor.submit(process_website, url, output_file): url for url in urls}
        for future in as_completed(future_to_url):
            try:
                future.result()
            except Exception as e:
                print(f"Ошибка при обработке сайта {future_to_url[future]}: {e}")
                logging.error(f"Ошибка при обработке сайта {future_to_url[future]}: {e}")


# Вызов основной функции с указанием файлов
input_file = "web.txt"
output_file = "transite.txt"
# Файл с результатами

process_websites(input_file, output_file)


