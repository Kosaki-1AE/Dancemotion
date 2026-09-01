import csv
import difflib

from flask import Flask, render_template, request, session

app = Flask(__name__)
app.secret_key = "secret!"  # セッションで履歴保存

# === 感性語辞書の読み込み ===
def load_dictionary(csv_path="感性語辞書_統合.csv"):
    dictionary = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dictionary[row["単語"]] = row
    return dictionary

kansengo_dict = load_dictionary()

# 音表記インデックス作成
def build_sound_index(dictionary):
    sound_index = {}
    for word, data in dictionary.items():
        sound_key = data.get("音表記", "").lower().strip()
        if sound_key:
            sound_index[sound_key] = data
    return sound_index

sound_dict = build_sound_index(kansengo_dict)

# 音類似語検索
def find_similar_sounds(query, sound_dict, cutoff=0.7):
    q = query.lower().strip()
    matches = difflib.get_close_matches(q, sound_dict.keys(), n=5, cutoff=cutoff)
    return [sound_dict[m] for m in matches if m in sound_dict]

# 感性の重みソート
emotion_weights = {"崇拝": 5, "哀愁": 4, "敬意": 4, "感嘆": 3, "肯定": 2, "不満": 1}
def sort_by_emotion_weight(entries):
    return sorted(entries, key=lambda x: emotion_weights.get(x.get("感情タグ", ""), 0), reverse=True)

# === メインルーティング ===
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    suggestions = []
    query = ""
    history = session.get("history", [])

    if request.method == "POST":
        query = request.form.get("word").strip().lower()
        result = sound_dict.get(query)
        if not result:
            suggestions = find_similar_sounds(query, sound_dict)
            suggestions = sort_by_emotion_weight(suggestions)

        # 履歴保存
        if query and query not in history:
            history.insert(0, query)
            if len(history) > 10:
                history = history[:10]
            session["history"] = history

    return render_template("index.html", result=result, query=query, suggestions=suggestions, history=history)

if __name__ == "__main__":
    app.run(debug=True)
