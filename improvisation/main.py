# main.py
import argparse
import sys

sys.path.append("./modules")

from train_scripts.camera_input import run_camera_session  # ※下にテンプレ置いた
# 既存の3つ＋将来用のagent
from train_scripts.camera_launcher_gui import main as gui_main
from train_scripts.serve_with_qr import main as server_main

# from agent import AgentConfig, TurnState, decide, update  # 使いたくなったら

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gui","server","camera"], default="gui")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--root", default="./web")
    ap.add_argument("--index", default="index.html")
    ap.add_argument("--tunnel", action="store_true")
    args = ap.parse_args()

    if args.mode == "gui":
        gui_main()  # ← 今のままでOK（後でサーバ起動ボタン生やせる）
    elif args.mode == "server":
        # serve_with_qr.py は自前のargparseを持ってるから環境に合わせて使うなら
        # そのまま `python serve_with_qr.py --tunnel` でもOK。
        # ここでは関数の再利用ではなく、素直に別プロセス起動でもアリ。
        server_main()
    elif args.mode == "camera":
        run_camera_session()  # ローカルWebカメラでSIM→HUD
    else:
        print("unknown mode")

if __name__ == "__main__":
    main()
