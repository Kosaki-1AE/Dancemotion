# vjepa_infer.py
# CPUでも即動く最小ラッパ。jepa 公式実装が見つかったら自動でそっちに切替。
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# ====== 1) 公式 V-JEPA を探す（任意） ======
# - リポジトリを clone 済みなら `pip install -e .` しておくと import 可能
# - 未導入なら ImportError → フォールバックに切替
_JEPA_AVAILABLE = False
try:
    # 例: facebookresearch/jepa を `pip install -e .` 済み想定
    # 実際のAPIはリポの構成に合わせてここを調整してください
    from jepa.models import build_model  # type: ignore
    _JEPA_AVAILABLE = True
except Exception:
    _JEPA_AVAILABLE = False
    build_model = None  # type: ignore


@dataclass
class VJEPAConfig:
    weight_path: Optional[str] = None   # 公式 ckpt の .pt など
    device: Optional[str] = None        # None→自動("cuda" or "cpu")
    image_size: int = 224               # 前処理の入力サイズ
    normalize_mean: float = 0.5
    normalize_std: float = 0.5
    return_embedding: bool = False      # Trueでベクトルも返す（重たくなる）


class _FallbackTiny(nn.Module):
    """
    フォールバック用の極小特徴抽出器（CPUで軽く動くやつ）。
    ※ 本物のV-JEPAが無い時でも配線確認できるようにする目的。
    """
    def __init__(self, in_ch=3, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, 2, 2), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x):  # x: (B,3,H,W) in [-1,1] 相当
        h = self.net(x).flatten(1)
        z = self.proj(h)
        z = nn.functional.normalize(z, dim=-1)
        return z  # (B, D)


class VJEPA:
    """
    インターフェースを固定:
      - __init__(weight_path, device)
      - embed(PIL.Image) -> dict(score: float, dim: int, backend: str, ...)
    """
    def __init__(self, weight_path: Optional[str] = None, device: Optional[str] = None, **kwargs):
        cfg = VJEPAConfig(weight_path=weight_path, device=device, **kwargs)
        self.cfg = cfg
        self.device = torch.device(cfg.device) if cfg.device \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 前処理（V-JEPA実装とスケールを合わせたい場合はここを後で調整）
        self.tf = T.Compose([
            T.Resize(cfg.image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(cfg.image_size),
            T.ToTensor(),
            T.Normalize(mean=[cfg.normalize_mean]*3, std=[cfg.normalize_std]*3),
        ])

        if _JEPA_AVAILABLE and (cfg.weight_path and os.path.isfile(cfg.weight_path)):
            # ===== 2) 本物のV-JEPAをロード =====
            self.backend = "vjepa"
            ckpt = torch.load(cfg.weight_path, map_location="cpu")
            # 公式の引数構造に合わせて build_model を呼ぶ（例）
            # 典型的には ckpt['args'] からモデルを再構築
            if isinstance(ckpt, dict) and "args" in ckpt:
                self.model = build_model(ckpt["args"])
                missing, unexpected = self.model.load_state_dict(ckpt.get("model", ckpt), strict=False)
            else:
                # もし state_dict そのものが来る場合
                self.model = build_model({})  # 必要に応じてダミーargs
                missing, unexpected = self.model.load_state_dict(ckpt, strict=False)
            self.model.to(self.device).eval()
            # 埋め込みの取り出しAPIは実装により異なるので、後段の _forward_embed で吸収
            self._dim = self._infer_dim()
        else:
            # ===== 3) フォールバック（軽いミニモデル）=====
            self.backend = "fallback"
            self.model = _FallbackTiny(embed_dim=128).to(self.device).eval()
            self._dim = 128

        # 推論の微ウォームアップ（CPUでも一瞬）
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.cfg.image_size, self.cfg.image_size, device=self.device)
            _ = self._forward_embed(dummy)

    def _infer_dim(self) -> int:
        # モデルから出る埋め込み次元を推定
        with torch.no_grad():
            x = torch.zeros(1, 3, self.cfg.image_size, self.cfg.image_size, device=self.device)
            z = self._forward_embed(x)
            return int(z.shape[-1])

    def _forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,H,W) in normalized scale.
        戻り値: (B, D) のL2正規化済み埋め込みを想定。
        """
        if self.backend == "vjepa":
            # 公式実装の「特徴取り出し」メソッドに合わせて書き換えポイント
            # 例：
            # z = self.model.forward_features(x)   # 実装のAPIにより名称が違う
            # z = pool/cls_token 等で (B, D) に集約
            # z = nn.functional.normalize(z, dim=-1)
            # return z
            # ---- 暫定: 公式APIが未確定の場合はエラーを避けるため平均プーリングで代替
            feats = self.model(x) if callable(self.model) else x
            if feats.ndim == 4:
                feats = nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
            elif feats.ndim > 2:
                feats = feats.mean(dim=tuple(range(1, feats.ndim)))
            feats = nn.functional.normalize(feats, dim=-1)
            return feats
        else:
            # フォールバック経路
            return self.model(x)

    @torch.no_grad()
    def embed(self, img: Image.Image) -> Dict[str, Any]:
        """
        1枚の画像からスコア等を返す。
        戻り値のフィールド（最低限）:
          - score: float（簡易的なノルム指標）
          - dim:   int   （埋め込み次元）
          - backend: "vjepa" or "fallback"
          - device:  "cpu" / "cuda"
          - (optionally) embedding: list[float]  # return_embedding=True のとき
        """
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device, non_blocking=True)
        z = self._forward_embed(x)   # (1, D)
        # スコア定義は用途に合わせて後で差し替え可能（とりあえずノルム）
        score = float(z.norm(dim=1).item())
        out = {
            "ok": True,
            "score": score,
            "dim": int(z.shape[-1]),
            "backend": self.backend,
            "device": str(self.device),
        }
        if self.cfg.return_embedding:
            out["embedding"] = z.squeeze(0).cpu().tolist()
        return out
