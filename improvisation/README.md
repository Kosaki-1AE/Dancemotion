
# 🎵 Improvisation AI - ImprovFormer

このプロジェクトは、**即興性（Improvisation）** と **空気感（Stillness）** を扱う AI モデル「ImprovFormer」の実装と、ダンスデータセット AIST++ を用いた空気感推論・可視化を目的としています。

---

## 📁 プロジェクト構成

```
improvisation/
├── aistdata++/                  # AIST++拡張データ処理モジュール
├── aist_json_sample/            # stillness/meaning付きJSONデータ
├── data/                        # responsibilityなどのスコアCSV
├── logs/                        # ログ用フォルダ（現在空）
├── models/
│   └── improvformer_model.pth   # 学習済みモデル重み
├── plots/
│   └── responsibility_score_plot.png
├── scripts/
│   ├── improvformer.py          # モデル定義
│   ├── train.py                 # 学習スクリプト
│   ├── predict_improvformer.py # 推論スクリプト
│   ├── infference_dummy.py     # ダミーデータ推論
│   ├── decision_engine.py      # Ψ-selector（決定エンジン）
│   ├── replay_buffer.py        # 履歴Stillness用バッファ
│   ├── action_features.py      # 空気感特徴抽出
│   └── utils.py, loader.py     # ユーティリティ
```

---

## ✅ 実装済み機能

- [x] ImprovFormer モデル定義と Positional Encoding with Stillness
- [x] 意味分類用 classifier 出力
- [x] 推論時の Attention 可視化対応
- [x] 空気感データ（Stillness）加算処理
- [x] Ψ-selector（即興決定エンジン）
- [x] AIST++ ベース構成対応

---

## 📌 実行の流れ

### 学習
```bash
python scripts/train.py
```

### 推論
```bash
python scripts/predict_improvformer.py
```

### ダミーデータで確認
```bash
python scripts/infference_dummy.py
```

### 意思決定トリガー例
```bash
python scripts/decision_engine.py
```

---

## 🔧 今後のToDo

- [ ] 意味ラベル付き教師データの自動生成
- [ ] attention-headの意味分類・空気感寄与分析
- [ ] AIST++ 実データを使った統合実験
- [ ] 空気感判定の定量評価（F1・混同行列など）

---

## ✨ 研究背景（ワイ理論・Stillness AI）

本プロジェクトは、ワイ理論における「Stillness（未確定のゆらぎ）」と「責任の矢（意思形成）」の概念を、Transformer系モデルに組み込んだAI構築を目指しています。

---

