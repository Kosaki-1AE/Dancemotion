# ChatGPT風：ダンス空気感と感情処理モデル（自然言語→感情→意味→応答スタイル）

import numpy as np


class EmotionProcessor:
    def __init__(self, context_memory=None):
        np.random.seed(42)  # 固定して毎回同じ重み
        self.memory = context_memory if context_memory is not None else np.zeros(5)
        self.W1 = np.random.randn(5, 5)
        self.W2 = np.random.randn(5, 5)

    def estimate_emotion(self, atmosphere_input):
        raw_emotion = np.dot(self.W1, atmosphere_input)
        return self.softmax(raw_emotion)

    def reframe_meaning(self, emotion_vec, frontal_control=0.5):
        adjusted_emotion = emotion_vec * (1 - frontal_control)
        meaning = np.tanh(np.dot(self.W2, adjusted_emotion) + self.memory)
        return meaning

    def generate_output(self, meaning_vec):
        return self.decode_response_style(meaning_vec)

    def decode_response_style(self, meaning_vec):
        tone = 'open' if meaning_vec[3] > 0 else 'closed'
        strength = float(np.clip(meaning_vec[0], -1, 1))
        gentleness = float(np.clip(1 - abs(meaning_vec[1]), 0, 1))
        abstraction = float(np.clip(meaning_vec[2], 0, 1))
        return {
            'tone': tone,
            'strength': strength,
            'gentleness': gentleness,
            'abstraction': abstraction
        }

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


class AIStyleResponder:
    def __init__(self):
        self.processor = EmotionProcessor(context_memory=np.zeros(5))

    def respond(self, user_text, frontal_control=0.4):
        atmosphere = self.infer_atmosphere_from_text(user_text)
        emotion = self.processor.estimate_emotion(atmosphere)
        meaning = self.processor.reframe_meaning(emotion, frontal_control=frontal_control)
        style = self.processor.generate_output(meaning)
        return self.generate_response(user_text, style), style

    def infer_atmosphere_from_text(self, text):
        text = text.lower()
        a1 = 0.8 if "悲しい" in text else 0.2
        a2 = 0.6 if "希望" in text else 0.2
        a3 = 0.4 if "静か" in text else 0.1
        a4 = 0.2 if "不安" in text else 0.1
        a5 = 0.1 if "強く" in text else 0.0
        return np.array([a1, a2, a3, a4, a5])

    def generate_response(self, text, style):
        tone_word = "やさしく" if style['tone'] == 'open' else "控えめに"
        strength_level = "強めに" if style['strength'] > 0.5 else "穏やかに"
        abstract_level = "抽象的に" if style['abstraction'] > 0.5 else "具体的に"
        return f"{tone_word}、{strength_level}、{abstract_level}返すね：「{text}」に対しての返答は…"


def inverse_atmosphere_from_response(style):
    a1 = style['abstraction'] * 0.6 + style['strength'] * 0.3
    a2 = style['gentleness'] * 0.5
    a3 = 1.0 if style['tone'] == 'open' else -1.0
    return np.array([a1, a2, a3, 0, 0])


def time_variant_response(text_series):
    responder = AIStyleResponder()
    responses = []
    for t, text in enumerate(text_series):
        response, _ = responder.respond(text)
        responses.append(response)
    return responses


# === 自動チューニング（方法2：ハイパーパラメータ探索） ===

def loss(pred, target):
    return (
        (pred['strength'] - target['strength'])**2 +
        (pred['gentleness'] - target['gentleness'])**2 +
        (pred['abstraction'] - target['abstraction'])**2 +
        (1.0 if pred['tone'] != target['tone'] else 0)
    )


def auto_tune_frontal_control(text, target_output):
    responder = AIStyleResponder()
    best_loss = float('inf')
    best_fc = 0.0
    best_style = None
    for fc in np.linspace(0.0, 1.0, 50):
        _, style = responder.respond(text, frontal_control=fc)
        l = loss(style, target_output)
        if l < best_loss:
            best_loss = l
            best_fc = fc
            best_style = style
    return best_fc, best_style


# === 使用例 ===

if __name__ == '__main__':
    text_input = "今日はなんだか静かで、少し希望がある日だね"
    responder = AIStyleResponder()
    print("[1回応答]:", responder.respond(text_input)[0])

    # 自動チューニングの実行
    target = {
        'tone': 'open',
        'strength': 0.7,
        'gentleness': 0.8,
        'abstraction': 0.6
    }
    best_fc, best_style = auto_tune_frontal_control(text_input, target)
    print("\n[自動チューニング結果] frontal_control:", round(best_fc, 3))
    print("出力スタイル:", best_style)

    # 時系列
    series = [
        "今日は少し悲しい",
        "でも希望が見えてきた",
        "静かな安心感がある",
        "やっぱり不安もあるけど前に進めそう"
    ]
    timeflow = time_variant_response(series)
    print("\n[時系列応答]:")
    for r in timeflow:
        print("  →", r)
