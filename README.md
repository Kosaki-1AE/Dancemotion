# Dancemotion Library

Dancemotion のファイルを管理するリポジトリです。

## 🚀How to Start

### ①Clone化してファイルのところに移動します

```bash
git clone https://github.com/Kosaki-1AE/Dancemotion.git

cd Dancemotion
```

### ② Add File & Commitで履歴を残します(自分がやったところまでを組み込むステップ)

```bash
git add . #「.」なら全て、指定の場合は指定のものにしてくだはい

git commit -m "内容"
```

### ③ Push（初回のみ -u） でリモートリポジトリに反映させます(これで「全員が見れる」というステップになります)

```bash
git push -u origin main
```

### ④ Pull で現在の最新状態を取得します(これで「最新の状態をローカルに保存する」というステップになります)

```bash
git fetch origin main #ここまでで一旦確認できる状態になります

git log HEAD..origin/main #これでローカルとリモートの差分を確認できます

git merge origin/main #ローカルのファイルが最新の状態になります
```

## 👥Author

Kosuke Sasaki Vidushan

[GitHub Profile](https://github.com/Kosaki-1AE)

## 🔗Discord Community

[![Join Discord](https://img.shields.io/badge/Discord-Join-blue?logo=discord)](https://discord.gg/tuhph8BxBF)