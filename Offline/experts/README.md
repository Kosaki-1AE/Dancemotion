
# Dance improvisation AI

このプロジェクトは、**即興性（Improvisation）** と **空気感(静動(Stillness in Motion))** を扱う AI モデル「ImprovFormer」の実装と、ダンスデータセット AIST++ を用いた空気感推論・可視化を目的として制作した。

---

## 📁 プロジェクト構成

```
improvisation/
├── README.md/                   # 今読んでるここのこと
├── aistdata++/                  # AIST++拡張データ処理モジュール
├── aist_json_sample/            # stillness/meaning付きJSONデータ
├── data/                        # responsibilityなどのスコアCSV
├── logs/                        # ログ用フォルダ（現在空）
├── models/
│   └── improvformer_model.pth   # 学習済みモデル重み
├── plots/
│   └── responsibility_score_plot.png
├── base_scripts/
│   ├── acts_core.py             # 活性化関数のレジストリ（正/負ペアをここで自動生成）
│   ├── anaylze.py               # 責任出力の集計・意思決定（実数版 & 表裏の表現手法としての複素版）
│   ├── complex_ops.py           # 複素責任ベクトルの生成/線形/活性/分解ユーティリティ
│   ├── contrib.py               # 正/負の出力を「効いた分」と「強さ」に分ける分解ロジック
│   ├── demo.py                  # 一連のデモ（活性ペアの比較出力＋Flowの挙動ログ）
│   ├── flow.py                  # Hashベースの方向づけ＋学習、イベント判定の“流れ”実装（実数/複素の両方）
│   ├── fluct.py                 # 心理的ゆらぎ（確率のノイズ化/閾値の揺らぎなど）
│   ├── init.py                  # 外部公開する最小インターフェイスをまとめて export
│   └── linops.py                # 線形変換（xW + b）の最小ユーティリティ
├── train_scripts/
│   ├── improvformer.py          # モデル定義
│   ├── train.py                 # 学習スクリプト
│   ├── predict_improvformer.py  # 推論スクリプト
│   ├── infference_dummy.py      # ダミーデータ推論
│   ├── decision_engine.py       # Ψ-selector（決定エンジン）
│   ├── replay_buffer.py         # 履歴Stillness用バッファ
│   ├── action_features.py       # 空気感特徴抽出
│   └── utils.py, loader.py      # ユーティリティ
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

## ✨ 研究背景（Stillness AI）

本プロジェクトは、著者自身のセルフフィードバックに基づく理論における「Stillness（未確定のゆらぎ）」と「責任の矢/責任ベクトル（意思形成）」の概念を、Transformer系モデルに組み込んだAI構築を目指すものである。

---

## 👥Author

Kosuke Sasaki