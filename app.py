import os
import threading
import uuid
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
from processing import process_job

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

JOBS = {}
LOCK = threading.Lock()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "Файл не выбран"}), 400

    if not f.filename.lower().endswith(".geojson"):
        return jsonify({"error": "Нужен GeoJSON-файл"}), 400

    job_id = uuid.uuid4().hex
    job_dir = DATA_DIR / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.geojson"
    f.save(input_path)

    with LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Задание поставлено в очередь",
            "error": None,
        }

    t = threading.Thread(
        target=run_job,
        args=(job_id, input_path, job_dir),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


def run_job(job_id, input_path, job_dir):
    def progress(value, message):
        with LOCK:
            if job_id in JOBS:
                JOBS[job_id]["progress"] = int(value)
                JOBS[job_id]["message"] = message

    try:
        with LOCK:
            JOBS[job_id]["status"] = "running"

        result = process_job(
            input_path=input_path,
            job_dir=job_dir,
            data_dir=DATA_DIR,
            progress=progress,
        )

        with LOCK:
            JOBS[job_id].update({
                "status": "done",
                "progress": 100,
                "message": "Готово",
                "result": result,
            })
    except Exception as e:
        traceback.print_exc()
        with LOCK:
            JOBS[job_id].update({
                "status": "error",
                "progress": 0,
                "message": "Ошибка",
                "error": str(e),
            })


@app.get("/api/status/<job_id>")
def status(job_id):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Задание не найдено"}), 404
        return jsonify(job)


@app.get("/api/result/<job_id>")
def result(job_id):
    with LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Задание не найдено"}), 404
    if job.get("status") != "done":
        return jsonify({"error": "Задание ещё не завершено"}), 409
    return jsonify(job["result"])


@app.get("/api/download/<job_id>/<kind>")
def download(job_id, kind):
    allowed = {
        "geojson": "result.geojson",
        "summary": "velo_demand_summary_3_periods.csv",
        "all": "velo_demand_all_periods.csv",
        "24_07": "velo_demand_24_07.csv",
        "25_07": "velo_demand_25_07.csv",
        "26_07": "velo_demand_26_07.csv",
    }
    filename = allowed.get(kind)
    if not filename:
        return jsonify({"error": "Неизвестный файл"}), 404

    path = DATA_DIR / "jobs" / job_id / filename
    if not path.exists():
        return jsonify({"error": "Файл не найден"}), 404

    return send_file(path, as_attachment=True, download_name=filename)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"Файл слишком большой. Максимум {MAX_UPLOAD_MB} МБ."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
