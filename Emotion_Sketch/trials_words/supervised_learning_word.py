# 教師データが貯まってから実行してちょうだいませ
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# ✅ JSONLファイルから読み込む（構造ラベルそのまま使用）
def load_structured_journal(jsonl_path="emotion_wordtrial.json"):
    data = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if "label" in item and "text" in item:
                    label = item["label"].strip()  # 例: "@性格"
                    text = item["text"].strip()
                    if label.startswith("@"):
                        data.append({"Label": label, "Text": text})
            except json.JSONDecodeError:
                print("⚠️ JSON 読み込み失敗行をスキップ")
    return pd.DataFrame(data)

# 📥 データ読み込み
df = load_structured_journal()

# ✨ TF-IDF ベクトル化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Text"])
y = df["Label"]

# 🎓 データ分割・学習・評価
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("=== 📊 評価レポート ===")
print(classification_report(y_test, y_pred))

# 🔍 新しい感情文に構造ラベルをつける推論
test_text = ["覚悟を決めて生きるってどこかで腹を括るってこと"]
test_vec = vectorizer.transform(test_text)
pred_label = clf.predict(test_vec)[0]
print(f"\n🧭 推論結果: 「{test_text[0]}」は → 構造ラベル「{pred_label}」")
