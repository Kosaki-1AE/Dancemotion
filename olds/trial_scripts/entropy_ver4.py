import cv2 as cv
import numpy as np
import sounddevice as sd
import queue, threading

# ========== Audio Stream ==========
q = queue.Queue()
def audio_callback(indata, frames, time, status):
    if status: print(status)
    q.put(indata.copy())

stream = sd.InputStream(channels=1, samplerate=16000, blocksize=512, callback=audio_callback)
stream.start()

def shannon_entropy_from_hist(h, eps=1e-12):
    p = h / (np.sum(h) + eps)
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))

def audio_entropy(block, bins=32):
    spec = np.abs(np.fft.rfft(block[:,0]))**2
    h, _ = np.histogram(spec, bins=bins, range=(0, np.max(spec)+1e-6))
    return shannon_entropy_from_hist(h)

# ========== Video ==========
cap = cv.VideoCapture(0)
assert cap.isOpened()

while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv.resize(frame, (640,360))
    gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # --- 映像側 Shannon (動きの揺らぎ) ---
    # ヒスト簡易: 輝度分布
    hist = cv.calcHist([gray],[0],None,[32],[0,256]).ravel()
    H_video = shannon_entropy_from_hist(hist)

    # --- 音声側 Shannon ---
    H_audio = None
    try:
        block = q.get_nowait()
        H_audio = audio_entropy(block)
    except queue.Empty:
        pass

    # --- 統合 ---
    if H_audio is not None:
        H_fused = 0.6*H_video + 0.4*H_audio
        cv.putText(frame, f"H_video:{H_video:.2f}", (20,40), cv.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv.putText(frame, f"H_audio:{H_audio:.2f}", (20,60), cv.FONT_HERSHEY_SIMPLEX,0.6,(0,200,200),2)
        cv.putText(frame, f"H_fused:{H_fused:.2f}", (20,80), cv.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
    else:
        cv.putText(frame, f"H_video:{H_video:.2f}", (20,40), cv.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv.imshow("Video+Audio Shannon", frame)
    if cv.waitKey(1)&0xFF==27: break

cap.release()
stream.stop()
cv.destroyAllWindows()
