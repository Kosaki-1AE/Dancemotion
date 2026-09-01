import json
import os
from datetime import datetime

BASE_DIR = "E:\\MyFeedbacks\\Student_Age"  # USBのルート
OUTPUT_PATH = "emotion_journal.json"

entries = []

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith(".txt"):
            label = os.path.basename(root)  # フォルダ名をラベルに
            file_path = os.path.join(root, file)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    entries.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "label": label,
                        "text": text
                    })

# 書き出し
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for entry in entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ {len(entries)} 件のジャーナルを変換 → {OUTPUT_PATH}")
