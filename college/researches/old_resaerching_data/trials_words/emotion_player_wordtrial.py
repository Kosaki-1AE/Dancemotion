import json
import os
from datetime import datetime

JSON_PATH = "emotion_journal.json"

def init_json():
    if not os.path.exists(JSON_PATH):
        with open(JSON_PATH, mode='w', encoding='utf-8') as _:
            pass  # 空ファイル作成だけでOK

def add_entry():
    label = input("📝 ラベルを入力してください（例: 覚悟, 比較, 気づきなど）: ").strip()
    print("🗒 感情ジャーナルの本文を入力してください（終了時は空行でEnter）:")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    text = "\n".join(lines)

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": label,
        "text": text
    }

    with open(JSON_PATH, mode='a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print("✅ JSONとして保存しました！")

def run_loop():
    print("🎙 感情ジャーナル記録システム（終了は Ctrl+C ）")
    while True:
        add_entry()
        print("-" * 40)

if __name__ == "__main__":
    init_json()
    run_loop()
