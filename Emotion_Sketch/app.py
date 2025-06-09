# ファイル: app.py　MNISTなんで一旦無視でおｋ
from flask import Flask, render_template, request

from mnist_journal_logger import train_model_with_journal

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result_text = None
    plot_base64 = None

    if request.method == "POST":
        epochs = int(request.form["epochs"])
        journal_texts = [request.form.get(f"journal{i+1}", "") for i in range(10)]
        plot_base64, result_text = train_model_with_journal(epochs, journal_texts)

    return render_template("index.html", result=result_text, image=plot_base64)

if __name__ == "__main__":
    app.run(debug=True)
