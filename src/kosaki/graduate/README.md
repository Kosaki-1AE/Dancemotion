# universal_mnist_cluster.py

## これは何をするか
入力されたデータ群（zip / csv / json / npy / xlsx / ディレクトリ）をまとめて読み込み、
各サンプルを **28x28 の MNIST 風形式** に統一したうえで、
**5〜10回程度のクラスタリング** を走らせ、
「**特徴の山**」と「**解釈**」を出力する Python プログラムです。

---

## インストール
```bash
pip install numpy pandas scikit-learn matplotlib openpyxl pyarrow
```

---

## 使い方
```bash
python universal_mnist_cluster.py --input ./data.zip --output ./out --clusters 8 --runs 8
```

ディレクトリを直接渡しても OK:
```bash
python universal_mnist_cluster.py --input ./my_data_folder --output ./out
```

---

## 主な出力
- `mnist_style_dataset.npz`
  - 28x28 相当の 784 次元に揃えたデータ
- `clustering_results.csv`
  - 各クラスタリング手法のスコア
- `consensus_labels.csv`
  - 複数回のクラスタ結果を統合したラベル
- `feature_peak_frequency.csv`
  - 全ランで頻出した特徴の山
- `final_report.json`
  - 総合レポート
- `final_report.txt`
  - 人間が読みやすい要約
- `*_representatives.png`
  - 各クラスタの代表サンプル画像

---

## ざっくり処理の流れ
1. 入力ファイルをまとめて読む
2. 数値特徴へ変換
3. 各サンプルを 784 次元へ補間して 28x28 化
4. PCA で圧縮
5. KMeans / GMM / Agglomerative / Birch / Spectral を seed 違い込みで 5〜10 本実行
6. 合意行列を作って consensus cluster を出す
7. 各クラスタの平均との差が大きい特徴を「特徴の山」として出す
8. その特徴をもとに自動解釈テキストを付ける

---

## 注意
- 完全に任意のデータから「意味のある解釈」を作るのは本質的に難しいです。
- このコードは **汎用パイプライン** として、
  まず「統一表現」「複数回クラスタリング」「山の抽出」「説明生成」までを通すことを目的にしています。
- 画像、音声、動画などまで含めてやるなら、前段の特徴抽出器を追加するのがおすすめです。
