# qstillness_adapter.py
# ----------------------------------------------------
# 「qstillness_engine.py」に“つなぐ”ための高レベルアダプタ。
# ・外部入力スケジュールで HumanSide.external_input を制御
# ・乱数シード固定
# ・Statevector からのサンプリング（測定相当）
# ・イベント用コールバック（LIGHTNING / SOUND window など）
# 依存: qiskit, numpy, dataclasses（標準）, typing（標準）

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
# あなたのエンジンを import
from qstillness_engine import Log, QuantumStillnessEngine, SimParams

# === コールバック用イベント名定義 ===
# on_step(t, ctx), on_lightning(t, bias, ctx), on_sound_begin(t, ctx), on_sound_end(t, ctx)
Callbacks = Dict[str, Callable[..., None]]

@dataclass
class RunConfig:
    params: Optional[SimParams] = None
    seed: Optional[int] = None
    # 外部入力スケジュール: t(ステップ) -> 0 or 1 を返す関数
    # None の場合はエンジンの内部ロジックに任せる
    external_input_schedule: Optional[Callable[[int], int]] = None
    # サンプリング（測定相当）設定: None なら測定しない
    sample_shots: Optional[int] = None       # 例: 1024
    sample_seed: Optional[int] = None        # サンプリング用シード
    # コールバック
    callbacks: Callbacks = field(default_factory=dict)

@dataclass
class StepTrace:
    t: int
    p_motion: float
    tension: float
    fear: float
    wonder: float
    external_input: int
    sampled_counts: Optional[Dict[str, int]] = None  # {'00':xxx, '01':...} など

@dataclass
class EpisodeTrace:
    steps: List[StepTrace] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    memory: List[str] = field(default_factory=list)
    params: SimParams = field(default_factory=SimParams)

class StillnessAdapter:
    """
    QuantumStillnessEngine を上位アプリから使いやすくする薄いアダプタ。
    - run_episode() で一括実行
    - コールバックで外側リアクション（ログ・可視化・音出しなど）
    """
    def __init__(self, config: Optional[RunConfig] = None):
        self.config = config or RunConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        self.engine = QuantumStillnessEngine(self.config.params)
        self._last_sound_flag = self.engine.side.external_input  # 0/1
        self.trace = EpisodeTrace(params=self.engine.params)

    # --- 内部: サンプリング（測定相当） ---
    def _sample_statevector(self, shots: int, seed: Optional[int]) -> Dict[str, int]:
        # Statevector.sample_counts を使って現在の量子状態を疑似測定
        # engine は「|q1 q0>」順なので、そのまま2bit文字列で返る
        sv = self.engine._sv
        counts = sv.sample_counts(shots=shots, seed=seed)
        # Dict[str, int] でそのまま返す（例: {'00':512,'01':256,'10':130,'11':126}）
        return dict(counts)

    # --- 内部: コールバック実行 ---
    def _cb(self, name: str, *args, **kwargs):
        if name in self.config.callbacks and callable(self.config.callbacks[name]):
            try:
                self.config.callbacks[name](*args, **kwargs)
            except Exception as e:
                # コールバック内での例外は握り潰して先へ（必要ならログ）
                pass

    # --- 1ステップ進める（エンジンの _step に直結） ---
    def _step(self, t: int):
        s = self.engine.side

        # 外部入力スケジュールが指定されていれば上書き
        if self.config.external_input_schedule is not None:
            try:
                s.external_input = int(self.config.external_input_schedule(t))
            except Exception:
                # 失敗したら無視して従来ロジックに任せる
                pass

        # 事前の external_input 値（サウンド窓の立ち上がり/終端検出用）
        prev_sound = self._last_sound_flag

        # === 実ステップ ===
        self.engine._step(t)

        # 稲妻イベント検出（エンジンが events に追記している）
        if len(self.engine.log.events) > 0 and (
            len(self.trace.events) == 0 or self.engine.log.events[-1] != self.trace.events[-1]
        ):
            # 直近のイベントを拾う
            ev = self.engine.log.events[-1]
            self.trace.events.append(ev)
            # 例: "[t=37] LIGHTNING (bias=0.92)" → bias 値を抽出して渡す
            bias = None
            try:
                if "bias=" in ev:
                    bias = float(ev.split("bias=")[1].split(")")[0])
            except Exception:
                pass
            self._cb("on_lightning", t, bias, ctx=self)

        # SOUND ウィンドウの立上/終端
        now_sound = self.engine.side.external_input
        if prev_sound == 0 and now_sound == 1:
            self._cb("on_sound_begin", t, ctx=self)
        elif prev_sound == 1 and now_sound == 0:
            self._cb("on_sound_end", t, ctx=self)
        self._last_sound_flag = now_sound

        # サンプリング（測定相当）
        sampled_counts = None
        if self.config.sample_shots is not None and self.config.sample_shots > 0:
            sampled_counts = self._sample_statevector(
                shots=self.config.sample_shots,
                seed=self.config.sample_seed
            )

        # ログ取り
        k = len(self.engine.log.p_motion) - 1
        step = StepTrace(
            t=t,
            p_motion=self.engine.log.p_motion[k],
            tension=self.engine.log.tension[k],
            fear=self.engine.log.fear[k],
            wonder=self.engine.log.wonder[k],
            external_input=self.engine.side.external_input,
            sampled_counts=sampled_counts
        )
        self.trace.steps.append(step)

        # on_step コールバック
        self._cb("on_step", t, ctx=self, step=step)

    # --- 公開 API: エピソード実行 ---
    def run_episode(self) -> EpisodeTrace:
        T = self.engine.params.T
        for t in range(T):
            self._step(t)
        # memory も持って帰る（エンジンが積んでいる詩的ログ）
        self.trace.memory = list(self.engine.log.memory)
        return self.trace


# ====== 使い方デモ（スクリプトとして実行した時だけ） ======
if __name__ == "__main__":
    # 1) 外部入力スケジュール例: 50〜70, 150〜170 は音の帰還を強制ON
    def ext_schedule(t: int) -> int:
        return 1 if (50 <= t < 70) or (150 <= t < 170) else 0

    # 2) 簡単なコールバック
    def on_lightning(t, bias, ctx):
        print(f"[adapter] ⚡ lightning at t={t}, bias={bias}")

    def on_sound_begin(t, ctx):
        print(f"[adapter] 🔊 sound window begin at t={t}")

    def on_sound_end(t, ctx):
        print(f"[adapter] 🔇 sound window end at t={t}")

    def on_step(t, ctx, step):
        # 例: 50 ステップ毎に進捗表示
        if t % 50 == 0:
            print(f"[adapter] t={t}, P(Motion)={step.p_motion:.3f}, ext={step.external_input}")

    config = RunConfig(
        params=SimParams(T=220),     # ステップ数などはここで調整
        seed=42,                     # 再現性
        external_input_schedule=ext_schedule,
        sample_shots=256,            # 状態の擬似測定（任意）
        sample_seed=7,
        callbacks={
            "on_lightning": on_lightning,
            "on_sound_begin": on_sound_begin,
            "on_sound_end": on_sound_end,
            "on_step": on_step
        }
    )

    adapter = StillnessAdapter(config)
    ep = adapter.run_episode()

    # 簡易の集計表示
    pm = np.array([s.p_motion for s in ep.steps])
    print(f"[adapter] mean P(Motion) = {pm.mean():.3f}")
    print("[adapter] first 3 events:", ep.events[:3])
    print("[adapter] first 2 memory:", ep.memory[:2])

    # もし可視化したければ、ここで matplotlib を使ってもOK
    # （君のエンジン側 __main__ と被らないように最低限だけ）
