import random
import time

import numpy as np
import sounddevice as sd

# === 初期化辞書類 ===
meaning_to_pattern = {}
pattern_to_meaning = {}
used_freqs = set()
conversation_log = []  # (発話者, 意味, [周波数列])

# === 拡張語彙セット ===
enhanced_vocab = [
    "了解", "進め", "止まれ", "繰り返せ", "異常あり", "話を変える", 
    "賛成する", "賛成しない", "もう一度言って", "何が起きた？", 
    "それは本当か？", "準備完了", "次の話題へ", "もうちょい詳しく",
    "やり直して", "分かりません", "それ面白い", "なるほど", "次に進もう",
    "感情は？", "もっと言って", "少し黙って", "ありがとう", "もう大丈夫"
]

# === 意味に対する応答辞書（文脈対話） ===
meaning_responses = {
    "何が起きた？": ["異常あり", "準備完了"],
    "それは本当か？": ["賛成する", "賛成しない"],
    "止まれ": ["了解", "次の話題へ"],
    "進め": ["準備完了", "了解"],
    "異常あり": ["止まれ", "繰り返せ"],
    "話を変える": ["次の話題へ", "それは本当か？"],
    "もう一度言って": ["繰り返せ"],
    "準備完了": ["進め"],
    "次の話題へ": ["それは本当か？", "何が起きた？"],
    "了解": ["進め"],
    "繰り返せ": ["もう一度言って"],
    "賛成する": ["了解"],
    "賛成しない": ["話を変える"]
}

# === 音を鳴らす関数 ===
def play_freq_sequence(freq_list, duration=0.4, pause=0.15, fs=44100):
    for freq in freq_list:
        try:
            t = np.linspace(0, duration, int(fs * duration), False)
            tone = np.sin(freq * 2 * np.pi * t)
            sd.play(tone, fs)
            sd.wait()
            time.sleep(pause)
        except Exception as e:
            print(f"[音再生エラー] {freq}Hz: {e}")

# === 周波数生成 ===
def generate_unique_freq():
    while True:
        freq = random.randint(200, 1000)
        if freq not in used_freqs:
            used_freqs.add(freq)
            return freq

# === 意味 → 周波数パターン生成 or 取得 ===
def get_or_generate_pattern(meaning):
    if meaning not in meaning_to_pattern:
        pattern_length = random.randint(2, 4)
        pattern = [generate_unique_freq() for _ in range(pattern_length)]
        meaning_to_pattern[meaning] = pattern
        pattern_to_meaning[tuple(pattern)] = meaning
    return meaning_to_pattern[meaning]

# === 意味生成（ルールベース + 語彙拡張） ===
def generate_new_meaning(prev_meaning=None):
    if prev_meaning in meaning_responses:
        return random.choice(meaning_responses[prev_meaning])
    else:
        return random.choice(enhanced_vocab)

# === AI発言ロジック（周波数ベース） ===
def ai_speak(prev_pattern, speaker):
    if prev_pattern is None:
        meaning = "何が起きた？"
    else:
        meaning = pattern_to_meaning.get(tuple(prev_pattern), "不明")

    next_meaning = generate_new_meaning(prev_meaning=meaning)
    next_pattern = get_or_generate_pattern(next_meaning)
    conversation_log.append((speaker, next_meaning, next_pattern))
    return next_meaning, next_pattern

# === 人間介入（意味→語彙に登録→音＋チャイム再生） ===
def human_intervention():
    print("\n=== 人間 介入モード ===")
    print("これまでの会話ログ：")
    for i, (who, meaning, pattern) in enumerate(conversation_log):
        pattern_str = ", ".join(f"{f}Hz" for f in pattern)
        print(f"{i+1:>2}: [{who}] {meaning} → [{pattern_str}]")

    user_meaning = input("次に送りたい意味を入力してください：").strip()
    if user_meaning not in enhanced_vocab:
        enhanced_vocab.append(user_meaning)

    pattern = get_or_generate_pattern(user_meaning)
    conversation_log.append(("人間", user_meaning, pattern))
    print(f"[人間] {user_meaning} → {[f'{f}Hz' for f in pattern]}")

    pattern_sorted = sorted(pattern)
    play_freq_sequence(pattern_sorted, duration=0.5, pause=0.2)

    confirmation_chime = [880, 660]
    play_freq_sequence(confirmation_chime, duration=0.25, pause=0.1)
    time.sleep(0.3)
    return pattern

# === 会話進行 ===
def start_pattern_conversation(rounds=10):
    current_pattern = None

    for i in range(rounds):
        print(f"\n--- Round {i+1} ---")

        meaning, current_pattern = ai_speak(current_pattern, "AI-A")
        print(f"[AI-A] {meaning} → {[f'{f}Hz' for f in current_pattern]}")
        play_freq_sequence(current_pattern)
        time.sleep(0.5)

        intervene = input("介入しますか？ (y/N): ").strip().lower()
        if intervene == "y":
            current_pattern = human_intervention()
        else:
            meaning, current_pattern = ai_speak(current_pattern, "AI-B")
            print(f"[AI-B] {meaning} → {[f'{f}Hz' for f in current_pattern]}")
            play_freq_sequence(current_pattern)
            time.sleep(0.5)

# === 実行 ===
if __name__ == "__main__":
    start_pattern_conversation(rounds=10)
