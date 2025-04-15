import asyncio
import json

import cv2
import numpy as np
import sounddevice as sd
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

# ---- 音声取得 ----
SAMPLING_RATE = 22050
BLOCK_SIZE = 1024
volume_level = 0.0

audio_buffer = np.zeros(BLOCK_SIZE)

def audio_callback(indata, frames, time, status):
    global volume_level, audio_buffer
    audio_buffer = indata[:, 0]
    volume_level = np.linalg.norm(audio_buffer) / frames

stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLING_RATE, blocksize=BLOCK_SIZE)
stream.start()

# ---- WebSocketエンドポイント ----
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    cap = cv2.VideoCapture(0)
    prev_frame = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ---- 動きの検出 ----
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_frame is None:
                prev_frame = blur
                await asyncio.sleep(0.01)
                continue

            frame_delta = cv2.absdiff(prev_frame, blur)
            motion_intensity = np.sum(frame_delta) / 1000000
            motion_intensity = np.clip(motion_intensity, 0, 1)
            prev_frame = blur

            # ---- 音量取得 ----
            vol = np.clip(volume_level, 0, 1)

            # ---- 空気感スコア ----
            score = (1 - vol) * (1 - motion_intensity)

            # ---- JSONで送信 ----
            data = {
                "motion": round(float(motion_intensity), 4),
                "volume": round(float(vol), 4),
                "stillness_score": round(float(score), 4)
            }

            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(0.03)  # 約30fps目安
    finally:
        cap.release()
