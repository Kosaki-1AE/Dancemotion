# sound_romaji_fixed.py
import pykakasi
from fugashi import Tagger

tagger = Tagger()
kakasi = pykakasi.kakasi()
kakasi.setMode("H", "a")
kakasi.setMode("K", "a")
kakasi.setMode("J", "a")
converter = kakasi.getConverter()

def extract_romaji(text):
    readings = []
    for word in tagger(text):
        try:
            reading = word.feature.kana
        except AttributeError:
            reading = word.surface
        romaji = converter.do(reading)
        readings.append(romaji)
    return ' '.join(readings)

if __name__ == "__main__":
    input_text = input("データを入力してください：")
    romaji_output = extract_romaji(input_text)
    print("🔤 ローマ字出力：", romaji_output)
