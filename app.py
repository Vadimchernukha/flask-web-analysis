from flask import Flask, render_template, request, send_file, jsonify
import threading
import logic  # Первая логика
import logic2  # Вторая логика
import time

app = Flask(__name__)

progress = 0
is_running = False
current_site = "Waiting..."
domains_loaded = 0
domains_after_stage1 = 0
domains_final = 0

input_file = "web.txt"  # Загруженный файл
output_file_stage1 = "transite.txt"  # Результат 1 этапа
output_file_final = "results.txt"  # Итоговый результат


def run_analysis():
    """
    Запускает двухэтапный анализ: сначала logic.py, затем logic2.py.
    """
    global progress, is_running, current_site, domains_after_stage1, domains_final
    is_running = True
    progress = 0

    with open(input_file, "r", encoding="utf-8") as f:
        sites = [line.strip() for line in f if line.strip()]

    total_sites = len(sites)
    if total_sites == 0:
        is_running = False
        return

    # 🔹 **Первый этап (общий анализ)**
    for index, site in enumerate(sites):
        if not is_running:
            break
        current_site = f"Stage 1: {site}"  # Обновляем текущий сайт
        logic.process_website(site, output_file_stage1)  # Первый анализ
        progress = int(((index + 1) / total_sites) * 50)  # Прогресс до 50%

    # Подсчитываем количество сайтов после 1 этапа
    with open(output_file_stage1, "r", encoding="utf-8") as f:
        domains_after_stage1 = len(f.readlines())

    # 🔹 **Второй этап (глубокий анализ)**
    with open(output_file_stage1, "r", encoding="utf-8") as f:
        stage1_results = [line.strip() for line in f if line.strip()]

    total_stage1 = len(stage1_results)
    for index, site in enumerate(stage1_results):
        if not is_running:
            break
        current_site = f"Stage 2: {site}"  # Обновляем текущий сайт
        logic2.process_website(site, output_file_final)  # Второй анализ
        progress = 50 + int(((index + 1) / total_stage1) * 50)  # Прогресс 50-100%

    # Подсчитываем финальное количество доменов
    with open(output_file_final, "r", encoding="utf-8") as f:
        domains_final = len(f.readlines())

    is_running = False
    progress = 100
    current_site = "Done!"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Загружает файл со списком сайтов и передает количество доменов в веб-интерфейс.
    """
    global domains_loaded
    file = request.files["file"]
    file.save(input_file)

    # ✅ Подсчет количества загруженных доменов
    with open(input_file, "r", encoding="utf-8") as f:
        domains_loaded = sum(1 for _ in f)  # Подсчитываем строки

    print(f"📂 Загружено доменов (после загрузки): {domains_loaded}")  # ✅ Проверка в терминале

    return jsonify({"message": "Файл загружен", "domains_loaded": domains_loaded})



@app.route("/get_domains_count")
def get_domains_count():
    """
    Возвращает количество загруженных доменов сразу после загрузки.
    """
    return jsonify({"domains_loaded": domains_loaded})


@app.route("/start_analysis")
def start_analysis():
    """
    Очищает файлы перед новой сессией и запускает анализ.
    """
    global is_running
    if not is_running:
        # ✅ Очищаем файлы перед началом нового анализа
        open(output_file_stage1, "w").close()  # Очищаем transite_stage1.txt
        open(output_file_final, "w").close()  # Очищаем transite_final.txt

        print("🗑️ Файлы очищены, начинаем новый анализ!")

        thread = threading.Thread(target=run_analysis)
        thread.start()

    return jsonify({"message": "Анализ начался"})



@app.route("/stop_analysis")
def stop_analysis():
    """
    Прерывает выполнение анализа.
    """
    global is_running
    is_running = False
    return "Анализ остановлен"


@app.route("/progress")
def get_progress():
    """
    Возвращает текущий прогресс выполнения анализа, текущий сайт и статистику.
    """
    return jsonify({
        "progress": progress,
        "current_site": current_site,
        "domains_loaded": domains_loaded,
        "domains_after_stage1": domains_after_stage1,
        "domains_final": domains_final
    })


@app.route("/download")
def download_file():
    """
    Отправляет пользователю файл с итоговыми результатами.
    """
    return send_file(output_file_final, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

