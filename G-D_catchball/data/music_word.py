from __future__ import annotations

import math
import random
import wave
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import MeCab
except Exception:
    MeCab = None


class MeaningSpace:
    def __init__(self, dim: int = 3, window: int = 3, lr: float = 0.05, seed: int = 42):
        self.dim = dim
        self.window = window
        self.lr = lr

        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.coords: Dict[int, List[float]] = {}
        self.word_count: Dict[str, int] = {}
        self.neighbor_variety: Dict[str, set[str]] = {}

        random.seed(seed)
        self.tagger = None
        if MeCab is not None:
            try:
                self.tagger = MeCab.Tagger("-r /etc/mecabrc")
            except Exception:
                self.tagger = None

    def tokenize(self, text: str) -> List[str]:
        text = str(text).strip()
        if not text:
            return []

        if self.tagger is not None:
            parsed = self.tagger.parse(text)
            if parsed is not None:
                tokens: List[str] = []
                for line in parsed.splitlines():
                    if line == "EOS" or not line:
                        continue
                    surface = line.split("\t")[0].strip()
                    if surface:
                        tokens.append(surface)
                if tokens:
                    return tokens

        if " " in text:
            return [tok for tok in text.split() if tok]
        return [ch for ch in text if ch.strip()]

    def get_id(self, word: str) -> int:
        if word not in self.word_to_id:
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word
            self.coords[idx] = [random.uniform(-0.1, 0.1) for _ in range(self.dim)]
        return self.word_to_id[word]

    def distance(self, id1: int, id2: int) -> float:
        v1 = self.coords[id1]
        v2 = self.coords[id2]
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    def expected_distance(self, token_distance: int) -> float:
        return 1.0 / token_distance

    def update_pair(self, id1: int, id2: int, token_distance: int) -> None:
        v1 = self.coords[id1]
        v2 = self.coords[id2]

        real_d = self.distance(id1, id2)
        exp_d = self.expected_distance(token_distance)

        if real_d == 0:
            for k in range(self.dim):
                v2[k] += random.uniform(-0.001, 0.001)
            real_d = self.distance(id1, id2)

        error = real_d - exp_d
        strength = 1.0 / token_distance

        word1 = self.id_to_word[id1]
        word2 = self.id_to_word[id2]

        self.neighbor_variety.setdefault(word1, set()).add(word2)
        self.neighbor_variety.setdefault(word2, set()).add(word1)

        var1 = len(self.neighbor_variety[word1])
        var2 = len(self.neighbor_variety[word2])
        var_weight = 1.0 / math.sqrt(max(1, var1 * var2))

        freq1 = self.word_count.get(word1, 1)
        freq2 = self.word_count.get(word2, 1)
        freq_weight = 1.0 / math.sqrt(max(1, freq1 * freq2))

        step = self.lr * strength * error * freq_weight * var_weight
        direction = [(b - a) / real_d for a, b in zip(v1, v2)]

        for k in range(self.dim):
            delta = step * direction[k]
            v1[k] += delta
            v2[k] -= delta

    def process_text(self, text: str) -> None:
        tokens = self.tokenize(text)
        if not tokens:
            return

        for tok in tokens:
            self.word_count[tok] = self.word_count.get(tok, 0) + 1

        token_ids = [self.get_id(tok) for tok in tokens]

        for i in range(len(token_ids)):
            for j in range(i + 1, min(len(token_ids), i + 1 + self.window)):
                id1 = token_ids[i]
                id2 = token_ids[j]
                td = j - i
                self.update_pair(id1, id2, td)

        self.center_all()
        self.normalize_all()

    def normalize_all(self) -> None:
        for idx in self.coords:
            v = self.coords[idx]
            norm = math.sqrt(sum(x * x for x in v))
            if norm > 0:
                self.coords[idx] = [x / norm for x in v]

    def center_all(self) -> None:
        n = len(self.coords)
        if n == 0:
            return

        mean = [0.0] * self.dim
        for v in self.coords.values():
            for i in range(self.dim):
                mean[i] += v[i]
        mean = [x / n for x in mean]

        for idx in self.coords:
            self.coords[idx] = [v - m for v, m in zip(self.coords[idx], mean)]

    def coord_to_freqs(self, coord: Sequence[float], base_freq: float = 220.0) -> List[float]:
        freqs = []
        for x in coord:
            freq = base_freq * (2 ** x)
            freqs.append(round(freq, 2))
        return freqs

    def word_to_music_info(self, word: str) -> Optional[Dict[str, Any]]:
        if word not in self.word_to_id:
            return None

        idx = self.word_to_id[word]
        coord = self.coords[idx]
        freqs = self.coord_to_freqs(coord)

        return {
            "word": word,
            "coord": [round(x, 4) for x in coord],
            "freqs": freqs,
        }

    def sentence_vector(self, text: str) -> Optional[List[float]]:
        tokens = self.tokenize(text)
        vecs = []

        for tok in tokens:
            if tok in self.word_to_id:
                idx = self.word_to_id[tok]
                vecs.append(self.coords[idx])

        if not vecs:
            return None

        return [sum(values) / len(values) for values in zip(*vecs)]

    def sentence_to_music_info(self, text: str) -> Optional[Dict[str, Any]]:
        vec = self.sentence_vector(text)
        if vec is None:
            return None

        freqs = self.coord_to_freqs(vec)

        return {
            "text": text,
            "vector": [round(x, 4) for x in vec],
            "freqs": freqs,
        }


@dataclass
class SynthConfig:
    sample_rate: int = 44100
    master_volume: float = 0.85
    note_duration: float = 0.42
    gap_duration: float = 0.05
    attack: float = 0.02
    release: float = 0.12


class MusicMapper:
    def __init__(self, engine: MeaningSpace, config: Optional[SynthConfig] = None):
        self.engine = engine
        self.config = config or SynthConfig()

    def coord_to_chord(self, coord: Sequence[float], base_freq: float = 220.0) -> List[float]:
        """
        3次元座標を3音コードへ変換
        - x0: ベース
        - x1: 中音
        - x2: 上音
        """
        if len(coord) < 3:
            padded = list(coord) + [0.0] * (3 - len(coord))
        else:
            padded = list(coord[:3])

        # 元コードの coord_to_freqs と同系統の変換
        base = base_freq * (2 ** padded[0])
        mid = base_freq * (2 ** padded[1]) * 1.25   # 長3度寄り
        top = base_freq * (2 ** padded[2]) * 1.5    # 完全5度寄り

        return [round(base, 2), round(mid, 2), round(top, 2)]

    def word_strength(self, word: str) -> float:
        freq = self.engine.word_count.get(word, 1)
        variety = len(self.engine.neighbor_variety.get(word, set()))
        # 出現頻度と文脈多様性を適当に混ぜて 0.25 ~ 1.0 へ
        s = 0.25 + min(0.75, 0.10 * math.log1p(freq) + 0.08 * math.log1p(variety + 1))
        return float(max(0.25, min(1.0, s)))

    def sentence_energy(self, text: str) -> float:
        tokens = self.engine.tokenize(text)
        if not tokens:
            return 0.4

        values = [self.word_strength(tok) for tok in tokens if tok in self.engine.word_to_id]
        if not values:
            return 0.4

        return float(max(0.25, min(1.0, sum(values) / len(values))))

    def envelope(self, n_samples: int) -> np.ndarray:
        cfg = self.config
        sr = cfg.sample_rate

        attack_n = max(1, int(cfg.attack * sr))
        release_n = max(1, int(cfg.release * sr))
        sustain_n = max(1, n_samples - attack_n - release_n)

        attack_env = np.linspace(0.0, 1.0, attack_n, endpoint=False)
        sustain_env = np.ones(sustain_n)
        release_env = np.linspace(1.0, 0.0, release_n, endpoint=True)

        env = np.concatenate([attack_env, sustain_env, release_env])

        if len(env) < n_samples:
            env = np.pad(env, (0, n_samples - len(env)))
        elif len(env) > n_samples:
            env = env[:n_samples]

        return env

    def synth_tone(
        self,
        freqs: Sequence[float],
        duration: float,
        amplitude: float = 0.5,
    ) -> np.ndarray:
        cfg = self.config
        sr = cfg.sample_rate
        n_samples = max(1, int(duration * sr))
        t = np.linspace(0, duration, n_samples, endpoint=False)

        signal = np.zeros_like(t)

        # 単純な倍音を少し足して、ちょい音楽っぽくする
        for freq in freqs:
            signal += 0.60 * np.sin(2 * np.pi * freq * t)
            signal += 0.25 * np.sin(2 * np.pi * (freq * 2.0) * t)
            signal += 0.15 * np.sin(2 * np.pi * (freq * 0.5) * t)

        signal /= max(1, len(freqs))
        signal *= self.envelope(n_samples)
        signal *= amplitude
        return signal.astype(np.float32)

    def silence(self, duration: float) -> np.ndarray:
        sr = self.config.sample_rate
        n_samples = max(1, int(duration * sr))
        return np.zeros(n_samples, dtype=np.float32)

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
        if peak <= 1e-8:
            return audio
        audio = audio / peak
        audio = audio * self.config.master_volume
        return audio.astype(np.float32)

    def text_to_music(
        self,
        text: str,
        base_freq: float = 220.0,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        入力文を順に処理して、
        各トークンを短いコードに変換してつなぐ
        """
        tokens = self.engine.tokenize(text)
        if not tokens:
            return np.array([], dtype=np.float32), []

        segments: List[np.ndarray] = []
        info_list: List[Dict[str, Any]] = []

        sentence_energy = self.sentence_energy(text)

        for tok in tokens:
            # 未学習語もここで空間に入れる
            if tok not in self.engine.word_to_id:
                self.engine.process_text(tok)

            idx = self.engine.word_to_id[tok]
            coord = self.engine.coords[idx]
            chord = self.coord_to_chord(coord, base_freq=base_freq)

            # 単語ごとの強さ
            amp = self.word_strength(tok) * sentence_energy
            amp = max(0.15, min(1.0, amp))

            tone = self.synth_tone(
                freqs=chord,
                duration=self.config.note_duration,
                amplitude=amp,
            )
            gap = self.silence(self.config.gap_duration)

            segments.append(tone)
            segments.append(gap)

            info_list.append({
                "token": tok,
                "coord": [round(x, 4) for x in coord],
                "chord": chord,
                "amplitude": round(float(amp), 4),
            })

        audio = np.concatenate(segments) if segments else np.array([], dtype=np.float32)
        audio = self.normalize_audio(audio)
        return audio, info_list

    def sentence_theme(
        self,
        text: str,
        duration: float = 1.8,
        base_freq: float = 220.0,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        文全体の平均ベクトルを1つのテーマ音にする
        """
        vec = self.engine.sentence_vector(text)
        if vec is None:
            return np.array([], dtype=np.float32), None

        chord = self.coord_to_chord(vec, base_freq=base_freq)
        amp = self.sentence_energy(text)

        audio = self.synth_tone(chord, duration=duration, amplitude=amp)
        audio = self.normalize_audio(audio)

        info = {
            "text": text,
            "vector": [round(x, 4) for x in vec],
            "theme_chord": chord,
            "amplitude": round(float(amp), 4),
        }
        return audio, info

    def save_wav(self, filename: str, audio: np.ndarray) -> None:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        pcm = np.int16(np.clip(audio, -1.0, 1.0) * 32767)

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.config.sample_rate)
            wf.writeframes(pcm.tobytes())


def print_music_info(info_list: List[Dict[str, Any]]) -> None:
    print("\n--- token music info ---")
    for item in info_list:
        print(
            f"{item['token']} | "
            f"coord={item['coord']} | "
            f"chord={item['chord']} | "
            f"amp={item['amplitude']}"
        )


def main() -> None:
    engine = MeaningSpace()
    mapper = MusicMapper(engine)

    print("文章入力（exit / quit で終了）")
    print("入力文を音楽化して WAV に保存します。")
    print("毎回:")
    print("  1. tokens_music.wav   -> 単語列を順につないだ音")
    print("  2. sentence_theme.wav -> 文全体の平均ベクトルから作るテーマ音")
    print("-" * 60)

    try:
        while True:
            text = input(">>> ").strip()
            if not text:
                continue
            if text.lower() in {"exit", "quit"}:
                break

            # 入力文で意味空間を更新
            engine.process_text(text)

            # 単語列ベースの音
            audio_tokens, token_info = mapper.text_to_music(text, base_freq=220.0)
            mapper.save_wav("tokens_music.wav", audio_tokens)

            # 文全体テーマ音
            audio_theme, theme_info = mapper.sentence_theme(text, duration=2.0, base_freq=220.0)
            mapper.save_wav("sentence_theme.wav", audio_theme)

            print_music_info(token_info)

            if theme_info is not None:
                print("\n--- sentence theme ---")
                print(
                    f"text={theme_info['text']}\n"
                    f"vector={theme_info['vector']}\n"
                    f"theme_chord={theme_info['theme_chord']}\n"
                    f"amplitude={theme_info['amplitude']}"
                )

            tokens = engine.tokenize(text)
            if tokens:
                last_tok = tokens[-1]
                last_info = engine.word_to_music_info(last_tok)
                print("\n--- last token info ---")
                print(last_info)

            print("\n保存完了:")
            print("  - tokens_music.wav")
            print("  - sentence_theme.wav")
            print("-" * 60)

    except KeyboardInterrupt:
        print("\n終了")


if __name__ == "__main__":
    main()