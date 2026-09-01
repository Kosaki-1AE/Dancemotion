
# -*- coding: utf-8 -*-
"""
bridge_adapter.py — base_scripts ⇄ train の橋渡しを標準化する薄いアダプタ層
目的：データ/モデル/判定を同一I/Fで差し替え可能にし、パイプラインを安定化。

使い方（例）:
from bridge_adapter import (
    BridgeConfig, AISTDataSource, ImprovFormerModel, BaseDeciderFactory,
    BridgeRunner,
)
cfg = BridgeConfig(data_folder="./aist_json_sample")
runner = BridgeRunner(
    data=AISTDataSource(cfg),
    model=ImprovFormerModel(cfg),
    decider=BaseDeciderFactory(cfg),
)
runner.run(log_csv="logs/bridge.csv", rb_out="data/replay.pkl")

必要ファイル：acts_core.py / analyze.py / flow.py / fluct.py / linops.py
               aist_loader.py / improvformer.py（or improvisation_dance.py）
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

import numpy as np
import torch
# train側
import train_scripts.aist_loader as AIST
import train_scripts.improvformer as IFM
# base側
from base_scripts.acts_core import get_activation
from base_scripts.flow import FlowState, make_ev_decider_core

try:
    import train_scripts.improvisation_dance as ID
except Exception:
    ID = None  # optional


# ======================
# 設定
# ======================
@dataclass
class BridgeConfig:
    # data/model
    data_folder: str = "./aist_json_sample"
    max_files: int = 8
    model_kind: str = "improvformer"   # "improvformer" | "stillness"
    input_dim: int = 51
    output_dim: int = 51
    num_heads: int = 2
    num_layers: int = 2
    d_ff: int = 256
    weights: Optional[str] = None
    device: str = "cpu"
    limit: Optional[int] = None

    # base 判定
    pos_activation: str = "silu"
    neg_activation: Optional[str] = None
    dim: int = 256
    tau: float = 1.0
    theta_init: float = 0.60
    ema: float = 0.05
    fluct_mode: str = "logit_gauss"
    fluct_sigma: float = 0.35
    center: str = "auto"


# ======================
# インターフェース
# ======================
class DataSource(Protocol):
    def __iter__(self) -> Iterable[torch.Tensor]:
        """各サンプルを [T, D] の torch.Tensor として順に返す"""
        ...

class Model(Protocol):
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """入力 [T, D] → 出力 [T, D] を返す"""
        ...

class Decider(Protocol):
    def step(self, token: str, base_delta: float) -> Dict[str, Any]:
        """p_hat/commit/theta_now を含む dict を返す"""
        ...


# ======================
# 具象実装（データ）
# ======================
class AISTDataSource:
    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.data = AIST.load_dataset_from_folder(cfg.data_folder, max_files=cfg.max_files)

    def __iter__(self) -> Iterable[torch.Tensor]:
        for i, seq in enumerate(self.data):
            if self.cfg.limit is not None and i >= self.cfg.limit:
                break
            yield seq


# ======================
# 具象実装（モデル）
# ======================
class ImprovFormerModel:
    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.model = IFM.ImprovFormer(
            input_dim=cfg.input_dim, output_dim=cfg.output_dim,
            num_meaning_classes=4, num_heads=max(1, cfg.num_heads),
            num_layers=max(1, cfg.num_layers), dim_feedforward=cfg.d_ff
        )
        if cfg.weights and os.path.exists(cfg.weights):
            state = torch.load(cfg.weights, map_location=torch.device(cfg.device))
            try:
                self.model.load_state_dict(state)
            except Exception:
                self.model.load_state_dict(state, strict=False)
        self.model.to(cfg.device).eval()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = seq.to(torch.float32).unsqueeze(0).to(self.cfg.device)  # [1,T,D]
        with torch.no_grad():
            out = self.model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
        return out.squeeze(0).detach().cpu()  # [T,D]


class StillnessModel:
    def __init__(self, cfg: BridgeConfig):
        assert ID is not None, "improvisation_dance.py が必要です"
        self.cfg = cfg
        self.model = ID.StillnessAI(dim=cfg.input_dim, num_heads=max(1, cfg.num_heads))
        if cfg.weights and os.path.exists(cfg.weights):
            state = torch.load(cfg.weights, map_location=torch.device(cfg.device))
            try:
                self.model.load_state_dict(state)
            except Exception:
                self.model.load_state_dict(state, strict=False)
        self.model.to(cfg.device).eval()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = seq.to(torch.float32).unsqueeze(0).to(self.cfg.device)
        with torch.no_grad():
            out = self.model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
        return out.squeeze(0).detach().cpu()


# ======================
# 具象実装（判定器）
# ======================
class BaseDeciderFactory:
    """base_scripts の Δ→p̂→commit 判定を提供"""
    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.flow = FlowState(dim=cfg.dim, theta0=cfg.theta_init, ema=cfg.ema)
        self._decider = None
        self._last_delta = 0.0

    def _ensure_decider(self):
        if self._decider is None:
            D = self.cfg.dim
            rng = np.random.default_rng(0)
            x = rng.normal(0, 1, size=(D,)).astype(np.float32)
            W = rng.normal(0, 0.15, size=(D, D)).astype(np.float32)
            b = rng.normal(0, 0.05, size=(D,)).astype(np.float32)
            pos = self.cfg.pos_activation
            neg = self.cfg.neg_activation or f"neg_{pos}"
            self._decider = make_ev_decider_core(
                x, W, b,
                pos_fn=get_activation(pos),
                neg_fn=get_activation(neg),
                tau=self.cfg.tau, center=self.cfg.center,
                fluct_mode=self.cfg.fluct_mode,
                fluct_kwargs={"sigma": self.cfg.fluct_sigma, "seed": 0},
                theta_init=self.cfg.theta_init,
            )

    def step(self, token: str, base_delta: float | None = None) -> Dict[str, Any]:
        self._ensure_decider()
        if base_delta is None:
            base_delta = self._last_delta
        out = self.flow.step(token, base_delta=base_delta, decider_fn=self._decider)
        # 次のΔ
        self._last_delta = 0.2 * base_delta + 0.1 * (1.0 if out["commit"] else -0.2)
        return out


# ======================
# ランナー
# ======================
class BridgeRunner:
    def __init__(self, data: DataSource, model: Model, decider: BaseDeciderFactory):
        self.data = data
        self.model = model
        self.decider = decider

    def run(self, log_csv: Optional[str] = None, rb_out: Optional[str] = None) -> Dict[str, Any]:
        writer = None
        fcsv = None
        if log_csv:
            os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)
            fcsv = open(log_csv, "w", newline="", encoding="utf-8")
            writer = csv.writer(fcsv)
            writer.writerow(["idx", "len_T", "commit", "p_hat", "theta", "mean_abs_out"])

        # RB は任意（遅延importで依存を軽く）
        rb = None
        if rb_out is not None:
            from replay_buffer_base import ReplayBufferBase  # type: ignore
            rb = ReplayBufferBase(capacity=10000)
            os.makedirs(os.path.dirname(rb_out) or ".", exist_ok=True)

        commits = 0
        total = 0

        for idx, seq in enumerate(self.data):
            pred = self.model.forward(seq)  # [T,D]
            mean_abs_out = float(pred.abs().mean().item()) if torch.is_tensor(pred) else float(np.abs(pred).mean())
            out = self.decider.step(f"seq#{idx}")

            if writer:
                writer.writerow([idx, int(seq.shape[0]), int(out["commit"]),
                                 f"{out['p_hat']:.6f}", f"{out['theta_now']:.6f}",
                                 f"{mean_abs_out:.6f}"])

            if rb is not None:
                responsibility = torch.tensor(out["p_hat"]).view(1)
                rb.push(seq, pred, responsibility)

            commits += int(out["commit"])
            total += 1

        if fcsv:
            fcsv.close()
        if rb is not None:
            rb.save(rb_out)  # type: ignore

        return {"commit_rate": commits / max(1, total), "seen": total}


# 便利ファクトリ（モデル選択）
def build_runner(cfg: BridgeConfig) -> BridgeRunner:
    data = AISTDataSource(cfg)
    if cfg.model_kind.lower() == "stillness":
        model = StillnessModel(cfg)
    else:
        model = ImprovFormerModel(cfg)
    decider = BaseDeciderFactory(cfg)
    return BridgeRunner(data=data, model=model, decider=decider)
