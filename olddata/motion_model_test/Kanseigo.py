# 感性語辞書：初代ChatGPT風 Webアプリ（Flask）

import csv

from flask import Flask, render_template, request

app = Flask(__name__)

# === 感性語辞書の読み込み ===
def load_dictionary(csv_path="感性語辞書_統合.csv"):
    dictionary = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dictionary[row["単語"]] = row
    return dictionary

# グローバルで辞書を読み込む
kansengo_dict = load_dictionary()

# === ルーティング ===
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    query = ""
    if request.method == "POST":
        query = request.form.get("word").strip()
        result = kansengo_dict.get(query)
    return render_template("index.html", result=result, query=query)


if __name__ == "__main__":
    app.run(debug=True)
