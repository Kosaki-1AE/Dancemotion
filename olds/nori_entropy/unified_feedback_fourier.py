# unified_feedback_fourier.py
# 時間軸×フーリエで「短期/長期/量子っぽい干渉」を一本化する最小実装
# 入力: 任意の時間列データ x(t) と 責任ベクトル r(t)
# 出力: Stillness/Motion/Interference 指標 + 次アクション

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

try:
    from scipy.signal import stft, get_window
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ====== Config ======
@dataclass
class Bands:
    low: Tuple[float, float]   # 低周波(長期) [Hz]
    mid: Tuple[float, float]   # 中期 [Hz]
    high: Tuple[float, float]  # 高周波(短期) [Hz]

@dataclass
class AlgoCfg:
    fs: float = 10.0           # サンプリング周波数 [Hz] (例: 0.1s刻みでログ→10Hz)
    win_sec: float = 8.0       # STFT窓サイズ [sec]
    hop_sec: float = 2.0       # STFTホップ [sec]
    bands: Bands = Bands(low=(0.0, 0.15), mid=(0.15, 0.6), high=(0.6, 2.0))
    # ↑帯域は用途で調整。長期ほど0に近い帯域を厚めに。

# ====== Data Structures ======
@dataclass
class Sample:
    t: float           # time [sec]
    x: float           # observed scalar (self score, reaction, PnL, etc.)
    r: float = 1.0     # responsibility weight in [0, +inf)
    ctx: str = "generic"  # "dance"/"invest"/"child"/...

@dataclass
class Metrics:
    stillness: float
    motion: float
    interference: float
    power_low: float
    power_mid: float
    power_high: float

# ====== Core ======
class UnifiedFeedbackFourier:
    def __init__(self, cfg: AlgoCfg):
        self.cfg = cfg
        self.short_gain = 1.0   # 短期側の学習ゲイン
        self.long_gain = 1.0    # 長期側の学習ゲイン
        self.bias = 0.0         # バイアス（環境/気分補正）
        # 可変しきい値（学習で動く）
        self.th_still = 0.55
        self.th_motion = 0.55

    def _to_uniform_ts(self, samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray]:
        """不規則サンプル→等間隔サンプリング列に補間（線形）。"""
        fs = self.cfg.fs
        if not samples:
            return np.array([]), np.array([])
        samples = sorted(samples, key=lambda s: s.t)
        t0, t1 = samples[0].t, samples[-1].t
        n = int(np.floor((t1 - t0) * fs)) + 1
        ts = t0 + np.arange(n) / fs

        t_arr = np.array([s.t for s in samples])
        x_arr = np.array([s.x for s in samples])
        r_arr = np.array([s.r for s in samples])

        # 責任込みの観測 y = r * x
        y_raw = r_arr * x_arr

        # 線形補間（境界はhold）
        y = np.interp(ts, t_arr, y_raw, left=y_raw[0], right=y_raw[-1])
        return ts, y

    def _band_power(self, f: np.ndarray, Pxx: np.ndarray, band: Tuple[float, float]) -> float:
        mask = (f >= band[0]) & (f < band[1])
        if not np.any(mask):
            return 0.0
        return float(np.trapz(Pxx[mask], f[mask]))

    def _stft_power(self, y: np.ndarray) -> Tuple[float, float, float]:
        fs = self.cfg.fs
        if SCIPY_OK:
            win = get_window("hann", int(self.cfg.win_sec * fs), fftbins=True)
            nperseg = len(win)
            noverlap = int((self.cfg.win_sec - self.cfg.hop_sec) * fs)
            f, t, Z = stft(y, fs=fs, window=win, nperseg=nperseg,
                           noverlap=noverlap if noverlap>0 else None, boundary=None)
            P = (np.abs(Z) ** 2).mean(axis=1)  # 時間平均パワー
        else:
            # 簡易FFT（全区間一括）。STFTがない環境用のフォールバック。
            Y = np.fft.rfft(y * np.hanning(len(y)))
            f = np.fft.rfftfreq(len(y), d=1.0/fs)
            P = (np.abs(Y) ** 2)

        low = self._band_power(f, P, self.cfg.bands.low)
        mid = self._band_power(f, P, self.cfg.bands.mid)
        high = self._band_power(f, P, self.cfg.bands.high)
        return low, mid, high

    def analyze(self, samples: List[Sample]) -> Optional[Metrics]:
        ts, y = self._to_uniform_ts(samples)
        if y.size < max(16, int(self.cfg.win_sec * self.cfg.fs / 2)):
            return None  # データ短すぎ

        low, mid, high = self._stft_power(y)
        total = low + mid + high + 1e-9
        still = low / total
        motion = high / total

        # 簡易“干渉”: 帯域間相関のproxy（正規化後の共分散から近似）
        # 雑に low/High の比と mid の存在で作る軽量スコア
        ratio = (min(low, high) / (max(low, high) + 1e-9))
        interference = float(0.5 * ratio + 0.5 * (mid / total))

        return Metrics(stillness=still, motion=motion, interference=interference,
                       power_low=low, power_mid=mid, power_high=high)

    # ====== フィードバック学習 ======
    def update_feedback(self,
                        reward_short: float,
                        reward_long: float,
                        metrics: Metrics):
        """
        reward_short: 直近の報い（例: 観客の歓声, 小利確, 今日の納得度）[-1..+1]
        reward_long : 長期の報い（例: 成長実感, 自立, 評判, 長期PF）[-1..+1]
        """
        # 学習ゲインの調整：どっちが効いてるかで配分を寄せる
        self.short_gain = np.clip(self.short_gain + 0.1 * reward_short, 0.2, 3.0)
        self.long_gain  = np.clip(self.long_gain  + 0.05 * reward_long , 0.2, 3.0)

        # しきい値の微調整：Still/Motion優勢ラインを学習
        self.th_still  = np.clip(self.th_still  + 0.02 * (reward_long  - 0.5), 0.45, 0.65)
        self.th_motion = np.clip(self.th_motion + 0.02 * (reward_short - 0.5), 0.45, 0.65)

        # バイアス：環境/気分の追従（干渉が高い=環境適合→バイアス下げ）
        self.bias = np.clip(self.bias + 0.05 * ((reward_short + reward_long)/2.0 - metrics.interference),
                            -0.5, 0.5)

    # ====== 意思決定（次アクション） ======
    def decide(self, m: Metrics) -> Dict[str, float]:
        """
        戻り値は “行動プリセット” の重み：
          attack: 攻め（出す/当てる/高密度）
          hold  : 待つ（引く/間合い/保有）
          sync  : 同期（キメ/共鳴/合わせ）
        """
        s = m.stillness + self.bias
        mo = m.motion - self.bias
        inter = m.interference

        # しきい値に基づく分岐 + 学習ゲインでスケーリング
        attack = float(np.clip((mo - self.th_motion) * 2.0, 0, 1) * self.short_gain)
        hold   = float(np.clip((s  - self.th_still ) * 2.0, 0, 1) * self.long_gain)
        sync   = float(np.clip(inter * 1.5, 0, 1))  # 干渉が高いと「合わせる」が有利

        # 正規化
        vec = np.array([attack, hold, sync]) + 1e-9
        vec = vec / vec.sum()
        return {"attack": float(vec[0]), "hold": float(vec[1]), "sync": float(vec[2])}

# ====== デモ（ダミーデータ） ======
if __name__ == "__main__":
    cfg = AlgoCfg()
    u = UnifiedFeedbackFourier(cfg)

    # ダミー時系列：高周波(短期) + 低周波(長期) + 雑音、責任は時間で変動
    rng = np.random.default_rng(42)
    T = 120.0  # 秒
    ts = np.arange(0, T, 1.0/cfg.fs)
    low_wave  = 0.8 * np.sin(2*np.pi*0.05*ts)      # 長期
    high_wave = 0.4 * np.sin(2*np.pi*0.9*ts)       # 短期
    noise = 0.15 * rng.standard_normal(len(ts))
    x = low_wave + high_wave + noise

    # 責任ベクトル（例：序盤は軽め→後半重め）
    r = np.linspace(0.7, 1.3, len(ts))

    samples = [Sample(t=float(t), x=float(xx), r=float(rr), ctx="demo")
               for t, xx, rr in zip(ts, x, r)]

    # 解析
    m = u.analyze(samples)
    if m is None:
        print("データ不足")
        raise SystemExit(0)

    print("[METRICS]",
          f"Stillness={m.stillness:.3f}, Motion={m.motion:.3f}, Interference={m.interference:.3f},",
          f"P_low={m.power_low:.2f}, P_mid={m.power_mid:.2f}, P_high={m.power_high:.2f}")

    # 直近の報い（例）：短期はややプラス、長期は強いプラス
    u.update_feedback(reward_short=+0.2, reward_long=+0.8, metrics=m)

    act = u.decide(m)
    print("[DECIDE]", act)
