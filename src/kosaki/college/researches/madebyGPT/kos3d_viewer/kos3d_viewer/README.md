# kos3d

Linuxのコマンドから開ける、無料の軽量3Dモデルビューアのたたき台です。

```bash
kos3d model.stl
kos3d model.3mf
kos3d model.obj
kos3d model.glb
```

## 対応予定/対応方針

このプロジェクトは Assimp を使うため、環境側の Assimp が読める形式を扱えます。

- STL: OK
- 3MF: Assimpのビルドが3MF対応ならOK
- OBJ: OK
- glTF/GLB: OK
- F3D / F3Z: Fusion 360のネイティブ形式なので直接読み込みは非対応。Fusion 360から STL または 3MF に書き出して読む想定。
- INO / HTML / Markdown: 3Dモデルではないのでビューア対象外。ただし将来的にシミュレータ情報として読み込む拡張は可能。

## Ubuntu / WSLg でのビルド

```bash
sudo apt update
sudo apt install -y build-essential cmake libglfw3-dev libassimp-dev libgl1-mesa-dev

mkdir -p build
cd build
cmake ..
cmake --build . -j
./kos3d ../sample.stl
```

## 操作

- 左ドラッグ: 回転
- ホイール: ズーム
- Q または Esc: 終了

## Box/Windows側ファイルをWSLから開く例

Windowsの `C:\Users\kos04\Box\...` はWSLではだいたい `/mnt/c/Users/kos04/Box/...` です。

```bash
./kos3d "/mnt/c/Users/kos04/Box/人力飛行機プロジェクト/各班/電気操縦班/2026年度_26代/おかだ/flight_stick.3mf"
```

パスに日本語や空白があるので、必ず `"..."` で囲むのが安全です。

## 次に足せる機能

- 寸法表示
- 重心/バウンディングボックス表示
- 複数モデル同時読み込み
- JSONでPico2W/ESP32/サーボの配置を読み込む
- UART/PWMの値で部品を動かす
