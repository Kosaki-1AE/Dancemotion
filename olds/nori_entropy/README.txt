Files:
- nori_onepass_unified.py  ... メイン。HUD・カメラ・音声・スコア・状態遷移。
- awareness_fast.py        ... NumPyのみの軽量エンジン（デフォ）。Aのみ、Tは1.0固定。
- awareness_torch.py       ... Torchの無意識エンジン（任意）。Self-AttnでA/Tを出す。

使い方:
1) まず軽量版で動かす（推奨）
   - nori_onepass_unified.py 内の USE_TORCH=False のまま起動
   - python nori_onepass_unified.py

2) 無意識（Torch）を試したい場合
   - pip install torch
   - nori_onepass_unified.py で USE_TORCH=True に変更
   - python nori_onepass_unified.py

ヒント:
- 重い場合は USE_AUDIO=False のまま（既定）
- 解像度やFarnebackパラメータ、矢印密度(step)は nori_onepass_unified.py 内を調整