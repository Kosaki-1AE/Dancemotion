# 日本語感性語辞書テンプレート＋初期データ＋自動収集＋AI補完（統合版）openai.api_keyでAPIキーを設定

import csv
import time

import openai
import requests
from bs4 import BeautifulSoup

# === ステップ①：テンプレート構造 ===
CSV_COLUMNS = [
    "単語", "カテゴリ", "時代", "感情タグ", "意味", "使われ方例", "類義語", "対義語", "備考"
]

# === ステップ②：初期リスト ===
INITIAL_WORDS = [
    ["やんごとない", "古語", "平安時代", "敬意", "高貴で尊い", "やんごとないお方に謁見する", "高貴", "卑しい", "敬語表現"],
    ["いとおかし", "古語", "平安時代", "感嘆", "とても風情がある、美しい", "いとおかしき景色かな", "風情がある", "無粋", "枕草子などに頻出"],
    ["チョベリバ", "死語", "1990年代", "不満", "とても悪い（チョー・ベリーバッド）", "今日はチョベリバな一日だった", "最悪", "最高", "女子高生俗語"],
    ["ナウい", "死語", "1980年代", "肯定", "流行していて今っぽい", "その服ナウいね！", "今風", "ダサい", "テレビ文化と共に流行"],
    ["エモい", "現代語", "2010年代", "哀愁", "感情を揺さぶる、懐かしさや切なさがある", "あの映画、めっちゃエモかった", "切ない", "無感情", "若者言葉"],
    ["尊い", "現代語", "2010年代", "崇拝", "あまりの愛しさ・神聖さに耐えられない気持ち", "推しが尊い…", "神々しい", "俗っぽい", "オタク界隈から普及"]
]

# === ステップ③：Wikipediaからの自動収集（例：死語一覧） ===
def scrape_words_from_wikipedia(url, keyword):
    res = requests.get(url)
    soup = BeautifulSoup(res.content, 'html.parser')
    words = []
    for li in soup.select("li"):
        text = li.get_text()
        if keyword in text:
            words.append(text.split("：")[0].strip())
    return list(set(words))

# === ステップ④：OpenAIで補完（APIキー必要） ===
openai.api_key = "YOUR_API_KEY"  # ←自分のAPIキーに置き換えて！

def complete_word_data(word):
    prompt = f"""
以下の日本語の単語に関して、カテゴリ（古語/死語/現代語など）、時代、感情タグ（ポジティブ/ネガティブ/敬意など）、意味、使用例、類義語、対義語、備考を日本語で出力してください。

単語: {word}
フォーマット: 単語,カテゴリ,時代,感情タグ,意味,使われ方例,類義語,対義語,備考
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        result = response['choices'][0]['message']['content']
        return result.strip().split(',')
    except Exception as e:
        print(f"エラー: {e}")
        return []

# === CSV出力 ===
def save_to_csv(filename, rows):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)
    print(f"CSVファイル保存完了：{filename}")


# === メイン実行 ===
if __name__ == "__main__":
    filename = "感性語辞書_統合.csv"

    # 初期データ追加
    all_rows = INITIAL_WORDS.copy()

    # Wikipediaからの自動収集（例）
    wiki_url = "https://ja.wikipedia.org/wiki/日本の死語一覧"
    wiki_words = scrape_words_from_wikipedia(wiki_url, "→")
    print("Wikipediaから抽出された候補数：", len(wiki_words))

    # GPTで補完（必要に応じて）
    for word in wiki_words[:10]:  # 時間かかるので10件限定
        print(f"補完中: {word}")
        data = complete_word_data(word)
        if len(data) == len(CSV_COLUMNS):
            all_rows.append(data)
        time.sleep(1.5)  # API制限対策

    save_to_csv(filename, all_rows)
