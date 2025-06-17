import shutil
import subprocess

# --- ① ffmpeg がシステムに存在するか確認 ---
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path is None:
    print("❌ ffmpeg が見つかりません。")
    print("🔧 以下の方法でインストールしてください：")
    print(" - Windows: https://www.gyan.dev/ffmpeg/builds/")
    print(" - macOS : brew install ffmpeg")
    print(" - Linux : sudo apt install ffmpeg")
    exit(1)
else:
    print(f"✅ ffmpeg は見つかりました: {ffmpeg_path}")

# --- ② 動作確認コマンドを実行 ---
try:
    result = subprocess.run(
        ["ffmpeg", "-version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode == 0:
        print("✅ ffmpeg は正常に動作しています！")
        print("🔎 バージョン情報：")
        print(result.stdout.split('\n')[0])
    else:
        print("⚠️ ffmpeg は見つかりましたが、動作に失敗しました。")
        print("詳細:")
        print(result.stderr)
except Exception as e:
    print("❌ ffmpeg の実行時にエラーが発生しました。")
    print(e)
