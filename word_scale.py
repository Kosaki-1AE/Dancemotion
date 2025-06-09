def map_consonants_to_scale(text):
    # 子音 → ドレミ音階のマッピング
    consonant = {
        'a': 'C',
        'b': 'C#',
        'c': 'D',
        'd': 'D#',
        'e': 'E',
        'f': 'F',
        'g': 'F#',
        'h': 'G',
        'i': 'G#',
        'j': 'A',
        'k': 'A#',
        'l': 'B',
        'm': 'C',
        'n': 'C#',
        'o': 'D',
        'p': 'D#',
        'q': 'E',
        'r': 'F',
        's': 'F#',
        't': 'G',
        'u': 'G#',
        'v': 'A',
        'w': 'A#',
        'x': 'B',
        'y': 'C',
        'z': 'C#'
    }

    notes = []
    for char in text.lower():
        if char in consonant:
            notes.append(consonant[char])
    return notes

# 使用例
if __name__ == "__main__":
    text = input("ローマ字の文章を入力してください：")
    print(map_consonants_to_scale(text))  # → ['C', 'F', 'F', 'D', 'A', 'F']
