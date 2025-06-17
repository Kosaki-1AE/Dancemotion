import csv
import os
from datetime import datetime

import pandas as pd

# サンプル用のテキストデータ（ユーザーが .txt に保存した形式と仮定）
txt_content = """
@性格
結婚するってのとただ付き合うってのの違いがあってさ。覚悟があるかないかっていう話になるじゃんか。
多分だけどね、死を前提とするか否かなの。
常に他人と比較するじゃんか、ワイら人間は。これを失くすにはどうすりゃいいんじゃろか。

@Mine研究
まず俺のフィードバックをそれぞれ数値化してみるべきだということに気づいてしまった。
これに辞書が付属されるとめっちゃそれっぽくなるはず。です。
"""

# テスト用に仮の .txt ファイルを作成
txt_path = "/mnt/data/sample_journal.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(txt_content)

# 読み込んで DataFrame 化する関数
def parse_txt_to_df(txt_path):
    data = []
    if not os.path.exists(txt_path):
        raise FileNotFoundError("指定されたパスの.txtファイルが見つかりません。")

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_label = None
    buffer = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("@"):
            if current_label and buffer:
                data.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Label": current_label,
                    "Text": "\n".join(buffer)
                })
                buffer = []
            current_label = stripped
        elif stripped:
            buffer.append(stripped)

    # 最後のセクションも保存
    if current_label and buffer:
        data.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Label": current_label,
            "Text": "\n".join(buffer)
        })

    return pd.DataFrame(data)

# 実行してDataFrame確認
df_result = parse_txt_to_df(txt_path)
df_result.to_csv("parsed_journal.csv", index=False, encoding="utf-8-sig")
print("✅ CSVに保存完了：parsed_journal.csv")
print(df_result.head())