import os
import uuid
from datetime import datetime

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'
LOG_FILE = 'saved_data/log.txt'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov', 'avi'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('saved_data', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_logs():
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(" | ")
                if len(parts) >= 4:
                    logs.append({
                        "id": parts[0],
                        "time": parts[1],
                        "text": parts[2],
                        "filename": parts[3]
                    })
    return logs

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        file = request.files.get("file")
        entry_id = str(uuid.uuid4())

        saved_name = "なし"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            saved_name = f"{timestamp}_{filename}"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], saved_name))

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{entry_id} | {now} | {text} | {saved_name}\n")

    logs = read_logs()
    return render_template("index.html", logs=logs)

@app.route("/delete_entry/<entry_id>")
def delete_entry(entry_id):
    if not os.path.exists(LOG_FILE):
        return redirect(url_for("index"))

    new_lines = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith(entry_id):
                new_lines.append(line)
            else:
                parts = line.strip().split(" | ")
                if len(parts) >= 4 and parts[3] != "なし":
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], parts[3])
                    if os.path.exists(filepath):
                        os.remove(filepath)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    return redirect(url_for("index"))

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
