import sqlite3

import pykakasi
from fugashi import Tagger

DB_PATH1 = "nuance_words.db"
DB_PATH2 = "rules.db"
tagger = Tagger()

def get_converter(mode="Hepburn"):
    kakasi = pykakasi.kakasi()
    kakasi.setMode("H", "a")
    kakasi.setMode("K", "a")
    kakasi.setMode("J", "a")
    kakasi.setMode("r", mode)
    return kakasi.getConverter()

def guess_nuance(text):
    conn = sqlite3.connect(DB_PATH1)
    c = conn.cursor()
    c.execute("SELECT word, nuance, score FROM nuance_words")
    rows = c.fetchall()
    conn.close()

    score = {}
    for word, nuance, weight in rows:
        if word in text:
            score[nuance] = score.get(nuance, 0) + weight

    return max(score, key=score.get)

def nuance_romaji_replace(romaji, nuance):
    conn = sqlite3.connect(DB_PATH2)
    c = conn.cursor()
    c.execute("SELECT romaji, replacement FROM rules WHERE nuance = ?", (nuance,))
    rows = c.fetchall()
    conn.close()

    for romaji_word, replacement in rows:
        romaji = romaji.replace(romaji_word, replacement)
    
    return romaji

def extract_romaji(text):
    nuance = guess_nuance(text)
    converter = get_converter("Kunrei")
    readings = []
    for word in tagger(text):
        try:
            reading = word.feature.kana
        except AttributeError:
            reading = word.surface
        romaji = converter.do(reading)
        romaji = nuance_romaji_replace(romaji, nuance)
        readings.append(romaji)
    return ' '.join(readings), nuance

if __name__ == "__main__":
    input_text = input("📝 日本語を入力してください：")
    romaji_output, guessed_nuance = extract_romaji(input_text)
    print("🌬️ 判定された空気感：", guessed_nuance)
    print("🔤 ローマ字出力：", romaji_output)
