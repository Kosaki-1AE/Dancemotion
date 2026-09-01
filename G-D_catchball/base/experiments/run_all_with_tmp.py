from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "input"
DATA_DIR = ROOT / "data"
TMP_DIR = ROOT / "tmp"
MERGE_DIR = DATA_DIR / "merge" / "array"

WORD_SCRIPT = ROOT / "data" / "realtime_3d_word.py"
MUSIC_SCRIPT = ROOT / "data" / "realtime_3d_music.py"
VIDEO_SCRIPT = ROOT / "data" / "realtime_3d_video.py"
MERGE_SCRIPT = ROOT / "data" / "merge" / "merge_modal_spaces.py"
INTEGRATOR_SCRIPT = ROOT / "data" / "merge" / "multimodal_integrator.py"

WORD_INPUT = INPUT_DIR / "word"
MUSIC_INPUT = INPUT_DIR / "musics"
VIDEO_INPUT = INPUT_DIR / "videos"

TMP_WORD = TMP_DIR / "word"
TMP_MUSIC = TMP_DIR / "music"
TMP_VIDEO = TMP_DIR / "video"


# =========================
# Utils
# =========================
def run(script, input_text=""):
    print(f"\n[RUN] {script}")
    subprocess.run(
        [sys.executable, str(script)],
        input=input_text,
        text=True,
        cwd=ROOT
    )


def latest_file(folder, exts):
    files = []
    for ext in exts:
        files += list(folder.glob(f"*{ext}"))
    if not files:
        return None
    return max(files, key=lambda x: x.stat().st_mtime)


def move_csv_to_tmp(names, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = ROOT / name
        if src.exists():
            shutil.move(src, dst / name)


def copy_tmp_to_root(tmp_dir):
    array_dir = DATA_DIR / "array"
    array_dir.mkdir(parents=True, exist_ok=True)

    for f in tmp_dir.glob("*.csv"):
        shutil.copy(f, array_dir / f.name)


def clean_tmp():
    for d in [TMP_WORD, TMP_MUSIC, TMP_VIDEO]:
        for f in d.glob("*.csv"):
            f.unlink()


# =========================
# Pipeline
# =========================
def run_word():
    txt = latest_file(WORD_INPUT, [".txt"])
    if not txt:
        raise Exception("word input not found")

    content = txt.read_text(encoding="utf-8")
    run(WORD_SCRIPT, content + "\nquit\n")

    move_csv_to_tmp([
        "word_token_summary.csv",
        "word_token_relationships.csv",
        "word_token_positions.csv"
    ], TMP_WORD)


def run_music():
    audio = latest_file(MUSIC_INPUT, [".wav", ".mp3"])
    if not audio:
        raise Exception("music not found")

    run(MUSIC_SCRIPT, str(audio) + "\n")

    move_csv_to_tmp([
        "token_summary.csv",
        "token_relationships.csv",
        "token_transitions.csv"
    ], TMP_MUSIC)


def run_video():
    video = latest_file(VIDEO_INPUT, [".mp4", ".mov", ".avi"])
    if not video:
        raise Exception("video not found")

    run(VIDEO_SCRIPT, str(video) + "\n")

    move_csv_to_tmp([
        "video_token_summary.csv",
        "video_token_relationships.csv",
        "video_token_transitions.csv",
        "video_frame_metadata.csv"
    ], TMP_VIDEO)


def run_merge():
    copy_tmp_to_root(TMP_WORD)
    copy_tmp_to_root(TMP_MUSIC)
    copy_tmp_to_root(TMP_VIDEO)

    run(MERGE_SCRIPT)


def run_integrator():
    copy_tmp_to_root(TMP_WORD)
    copy_tmp_to_root(TMP_MUSIC)
    copy_tmp_to_root(TMP_VIDEO)

    run(INTEGRATOR_SCRIPT)


def main():
    print("=== START ===")

    run_word()
    run_music()
    run_video()

    run_merge()
    run_integrator()

    clean_tmp()

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()