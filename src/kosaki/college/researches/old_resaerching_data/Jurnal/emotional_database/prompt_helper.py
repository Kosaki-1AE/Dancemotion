def classify_air(user_text):
    if any(word in user_text for word in ["夜", "静か", "雨", "眠", "淡い", "影", "遠い"]):
        return "静けさ、余韻、孤独"
    elif any(word in user_text for word in ["笑", "陽", "朝", "光", "走", "未来", "希望"]):
        return "明るさ、開放、希望"
    elif any(word in user_text for word in ["揺れ", "曖昧", "風", "雲", "中間"]):
        return "曖昧、揺れ、曇り"
    else:
        return "不定、流動、未知"

def generate_mock_poem(tags):
    if "静けさ" in tags:
        return "風が通り過ぎたあと、心だけが残った。"
    elif "明るさ" in tags:
        return "光が跳ねて、未来の音がした。"
    elif "曖昧" in tags:
        return "この曖昧な空の色も、誰かの心に似ている。"
    else:
        return "名もなき感情が、まだ形を探している。"
