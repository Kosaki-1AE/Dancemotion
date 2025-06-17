def map_consonants_to_scale(text):
    # 子音 → ドレミ音階のマッピング これを色々変更していきます、はい。
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
        'ss': 'F#',
        't': 'G',
        'u': 'G#',
        'v': 'A',
        'w': 'A#',
        'x': 'B',
        'y': 'C',
        'z': 'C#',
        'ar': 'M',
        'ee': 'M',
        'ea': 'M',
        'oo': 'M',
        'dg': 'M',
        'ch': 'M',
        'tch': 'F#',
        'th': 'M',
        'ts': 'F#',
        'sh': 'M',
        'zh': 'M',
        'si': 'M',
        'ng': 'M',
        'aw': 'M',
        'or': 'M',
        'ay': 'M',
        'ei': 'M',
        'igh': 'M',
        'ow': 'M',
        'ou': 'M',
        'oi': 'M',
        'oy': 'M'
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
