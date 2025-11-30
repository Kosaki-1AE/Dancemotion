# -*- coding: utf-8 -*-
"""
Nori AV Shannon++ (Integrated)
- Video+Audio Shannon Entropy (robust, EMA, dynamic norm)
- State machine: STILL/MOTION + spike → event snapshots(JSON+PNG)
- CuriosityAgent: pick high-motion tiles → ROI save & Q/A
- Non-blocking chat: stdin thread (type in terminal: 'what' or 'answer=猫')
- JP HUD via Pillow(Meiryo) if available, fallback to cv2.putText

Keys:
  Esc : quit
  A/Z : audio weight -/+
  [ / ] : STILL_THR_ENTER down/up
  { / } : STILL_THR_EXIT  down/up
  C : print latest curiosity question (same as typing 'what')
"""

import os, sys, json, time, queue, threading
from collections import deque
from dataclasses import dataclass

import numpy as np
import cv2 as cv
import sounddevice as sd

# ========== Prefs ==========
PREF_IN_NAME = ""   # e.g., "Microphone (Realtek". empty -> default
SR = 16000
BLOCK = 512
SIZE = (640, 360)

SNAP_DIR = "eventsnaps"
CURI_DIR = "curiosity_snaps"
os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(CURI_DIR, exist_ok=True)

# ========== JP Text (optional Pillow) ==========
try:
    from PIL import Image, ImageDraw, ImageFont
    JP_FONT = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 20)  # adjust path on Linux/Mac
    def draw_text(frame_bgr, text, xy=(10, 170), fill=(255,255,255)):
        img = Image.fromarray(cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img); d.text(xy, text, font=JP_FONT, fill=fill)
        return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    JP_OK = True
except Exception:
    def draw_text(frame_bgr, text, xy=(10, 170), fill=(255,255,255)):
        cv.putText(frame_bgr, text, xy, cv.FONT_HERSHEY_SIMPLEX, 0.55, fill, 1, cv.LINE_AA)
        return frame_bgr
    JP_OK = False

# ========== Utils ==========
def shannon_entropy_from_hist(h, eps=1e-12):
    s = float(np.sum(h)); 
    if s <= eps: return 0.0
    p = h / (s + eps); p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))

def ema(prev, x, a): 
    return float(x) if prev is None else float(a*x + (1-a)*prev)

def normalize_dyn(x, lo, hi, eps=1e-9):
    return float(np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0))

# ========== Audio ==========
aq = queue.Queue()
last_audio_err = ""
audio_alive = False

def pick_input_device(name_part: str):
    try:
        devs = sd.query_devices()
        if not name_part: return None
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0 and name_part.lower() in d["name"].lower():
                return i
    except Exception as e:
        print("[query_devices ERR]", e)
    return None

def start_audio_stream():
    global audio_alive, last_audio_err
    try:
        in_idx = pick_input_device(PREF_IN_NAME)
        if in_idx is not None:
            sd.default.device = (in_idx, None)
        sd.default.samplerate = SR
        stream = sd.InputStream(
            channels=1, samplerate=SR, blocksize=BLOCK,
            callback=lambda indata, frames, time_info, status: (
                aq.put(indata.copy()),
                sys.stderr.write(f"[AUDIO status] {status}\n") if status else None
            )
        )
        stream.start()
        audio_alive = True; last_audio_err = ""
        return stream
    except Exception as e:
        audio_alive = False; last_audio_err = str(e)
        print("[AUDIO start ERR]", e)
        return None

def pop_latest(q: queue.Queue):
    latest = None
    try:
        while True:
            latest = q.get_nowait()
    except queue.Empty:
        pass
    return latest

def audio_entropy(block, bins=32, clip_p=99.5):
    x = block[:, 0].astype(np.float32)
    x = x - np.mean(x)
    win = np.hanning(len(x)).astype(np.float32)
    X = np.abs(np.fft.rfft(x * win)) ** 2
    hi = np.percentile(X, clip_p) + 1e-6
    h, _ = np.histogram(np.clip(X, 0, hi), bins=bins, range=(0, hi))
    return shannon_entropy_from_hist(h)

# ========== Curiosity ==========
class CuriosityAgent:
    def __init__(self, save_dir=CURI_DIR, k=2, hv_low=0.45, hv_high=1.15, sim_thresh=0.92, cooldown=0.9):
        self.save_dir = save_dir; os.makedirs(save_dir, exist_ok=True)
        self.k=k; self.hv_low=hv_low; self.hv_high=hv_high
        self.sim_thresh=sim_thresh; self.cooldown=cooldown
        self.last_feats=deque(maxlen=40); self.questions=deque(maxlen=30)
        self.last_fire_t=0.0; self.marks=deque(maxlen=120)

    def _push_mark(self,x,y,size,mode="what"):
        self.marks.append((int(x),int(y),int(size),time.time(),mode))

    def get_marks(self,decay=3.0):
        now=time.time()
        return [(x,y,size,mode) for (x,y,size,ts,mode) in self.marks if (now-ts)<decay]

    @staticmethod
    def _roi_feats(img):
        small=cv.resize(img,(32,32))
        hist=cv.calcHist([small],[0],None,[32],[0,256]).flatten()
        hist/= (np.sum(hist)+1e-9)
        return np.concatenate([small.flatten()/255.0,hist]).astype(np.float32)

    @staticmethod
    def _cos(a,b):
        na,nb=np.linalg.norm(a),np.linalg.norm(b)
        if na<1e-9 or nb<1e-9: return 0.0
        return float(np.dot(a,b)/(na*nb))

    def _novel(self,feat):
        if not self.last_feats: return True
        return (max(self._cos(feat,f) for f in self.last_feats)<self.sim_thresh)

    def step(self, frame_bgr, gray, flow, H_video_ema, e_video):
        now=time.time()
        if not(self.hv_low<=(H_video_ema or 0.0)<=self.hv_high): return None
        if e_video<0.08: return None
        if (now-self.last_fire_t)<self.cooldown: return None
        fx,fy=flow[...,0],flow[...,1]; mag=np.sqrt(fx*fx+fy*fy)
        H,W=mag.shape; step=32; cands=[]
        for y in range(0,H-step,step):
            for x in range(0,W-step,step):
                tile=mag[y:y+step,x:x+step]; v=float(np.max(tile))
                yy,xx=np.unravel_index(np.argmax(tile),tile.shape)
                cands.append((v,x+xx,y+yy,step))
        if not cands: return None
        cands.sort(reverse=True); picks=cands[:self.k]
        for _,cx0,cy0,sz in picks: self._push_mark(cx0,cy0,sz,"what")
        for _,cx,cy,sz in picks:
            x0=max(0,cx-64); x1=min(W,cx+64); y0=max(0,cy-64); y1=min(H,cy+64)
            roi=gray[y0:y1,x0:x1]
            if roi.size<64*64: continue
            feat=self._roi_feats(roi)
            if not self._novel(feat): continue
            self._push_mark(cx,cy,sz,"what")
            path=f"{self.save_dir}/roi_{int(time.time())}_{cx}x{cy}.png"
            cv.imwrite(path,frame_bgr[y0:y1,x0:x1])
            self.last_feats.append(feat)
            self.questions.append((path,(x0,y0,x1,y1),now,f"これは何？@({cx},{cy})"))
            self.last_fire_t=now; return self.questions[-1][3]
        return None

    def latest_question(self): return self.questions[-1] if self.questions else None
    def answer_latest(self,ans_text):
        if not self.questions: return "（質問なし）"
        path,bbox,ts,qtext=self.questions[-1]
        base,ext=path.rsplit(".",1); new_path=f"{base}__{ans_text}.{ext}"
        os.rename(path,new_path)
        if self.marks:
            x,y,size,t,mode=self.marks[-1]; self.marks[-1]=(x,y,size,t,"accepted")
        return f"そういうものかァ → {new_path}"

# ========== Non-blocking Chat ==========
class ChatIO:
    def __init__(self):
        self.q = queue.Queue(); self._alive = True
        self._thr = threading.Thread(target=self._run, daemon=True); self._thr.start()
        print("[CHAT] ここに入力して Enter → 映像は止まりません。例: what / answer=猫")
    def _run(self):
        while self._alive:
            try: s = input()
            except EOFError: break
            if not s: continue
            self.q.put(s.strip())
    def poll(self):
        try: return self.q.get_nowait()
        except queue.Empty: return None
    def stop(self): self._alive = False

class ChatAgent:
    def __init__(self, persona="univ_girl"):
        self.persona=persona
        self.last="準備OK〜 'what' で最新の質問、'answer=...' で回答してね"
    def _style(self, t):
        if self.persona=="univ_girl": return f"{t} って感じかな〜"
        return t
    def reply(self, text, ctx):
        curi = ctx["curiosity"]
        if "answer=" in text:
            return self._style(curi.answer_latest(text.split("=",1)[1].strip()))
        if text.lower()=="what":
            q=curi.latest_question(); return self._style(q[3] if q else "まだ質問ないかも〜")
        return self._style("お題ちょーだい")

# ========== Main ==========
def main():
    cap = cv.VideoCapture(0)
    assert cap.isOpened(), "Camera not opened"
    audio_stream = start_audio_stream()
    chat_io = ChatIO()
    chat = ChatAgent()
    curiosity = CuriosityAgent()

    # Shannon states
    H_video_ema=None; H_audio_ema=None; H_fused_ema=None
    v_min, v_max = 1e9, -1e9
    a_min, a_max = 1e9, -1e9
    w_video, w_audio = 0.6, 0.4
    fps = 0.0; last_fps_t = time.time(); frame_cnt = 0
    last_audio_t = time.time()

    # State machine
    STILL_THR_ENTER = 0.28
    STILL_THR_EXIT  = 0.40
    STILL_MIN_DUR   = 0.8
    SPIKE_THR       = 0.20
    state = "MOTION"; state_since = time.time(); last_H = None
    event_id = 0

    def snap(frame_bgr, meta: dict):
        nonlocal event_id
        event_id += 1
        ts = time.time()
        png = os.path.join(SNAP_DIR, f"{event_id:06d}.png")
        jso = os.path.join(SNAP_DIR, f"{event_id:06d}.json")
        cv.imwrite(png, frame_bgr)
        with open(jso, "w", encoding="utf-8") as f:
            m = dict(meta); m["ts"]=ts; m["image"]=os.path.basename(png)
            json.dump(m, f, ensure_ascii=False, indent=2)
        print(f"[EVT] saved {png} :: {m['type']}")

    def on_event(evtype: str, payload: dict, frame_bgr):
        snap(frame_bgr, {"type": evtype, **payload})

    prev_gray=None; last_chat_t=0

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv.resize(frame, SIZE)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # ---- VIDEO Shannon (luma hist) ----
            hist = cv.calcHist([gray],[0],None,[32],[0,256]).ravel()
            H_video = shannon_entropy_from_hist(hist)
            H_video_ema = ema(H_video_ema, H_video, 0.25)
            v_min = min(v_min * 0.995 + H_video_ema * 0.005, H_video_ema)
            v_max = max(v_max * 0.995 + H_video_ema * 0.005, H_video_ema)
            Hv_n  = normalize_dyn(H_video_ema, v_min, v_max)

            # ---- AUDIO ----
            blk = pop_latest(aq); Ha_n=None
            if blk is not None:
                try:
                    Ha_raw = audio_entropy(blk, bins=32, clip_p=99.5)
                    H_audio_ema = ema(H_audio_ema, Ha_raw, 0.35)
                    a_min = min(a_min * 0.995 + H_audio_ema * 0.005, H_audio_ema)
                    a_max = max(a_max * 0.995 + H_audio_ema * 0.005, H_audio_ema)
                    Ha_n = normalize_dyn(H_audio_ema, a_min, a_max)
                    last_audio_t = time.time()
                except Exception as e:
                    pass
            else:
                if time.time() - last_audio_t > 2.0:
                    if audio_stream is not None:
                        try: audio_stream.stop(); audio_stream.close()
                        except Exception: pass
                    start_audio_stream(); last_audio_t=time.time()

            # ---- FUSE ----
            Hf = Hv_n if Ha_n is None else (w_video*Hv_n + w_audio*Ha_n)
            H_fused_ema = ema(H_fused_ema, Hf, 0.2)
            Hn = float(np.clip(H_fused_ema if H_fused_ema is not None else Hf, 0.0, 1.0))

            # ---- Optical Flow for Curiosity ----
            if prev_gray is None:
                prev_gray = gray.copy()
            flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5,3,15,3,5,1.2,0)
            prev_gray = gray.copy()

            # Curiosity fires near balanced motion regime
            qtext = curiosity.step(frame, gray, flow, H_video_ema or 0.0, 0.1)
            if qtext: print("[Q]", qtext)

            for x,y,size,mode in curiosity.get_marks():
                color = (255,0,0) if mode=="what" else (128,128,128)
                label = "?" if mode=="what" else "…"
                cv.circle(frame,(x,y),max(3,size//10),color,-1)
                cv.putText(frame,label,(x+6,y-6),cv.FONT_HERSHEY_SIMPLEX,0.6,color,2)

            # ---- State machine & events ----
            if last_H is not None and abs(Hn - last_H) >= SPIKE_THR:
                on_event("spike", {"dH": round(float(Hn-last_H),3), "H": round(Hn,3)}, frame.copy())
            last_H = Hn

            if state == "MOTION":
                if Hn <= STILL_THR_ENTER:
                    state="STILL"; state_since=time.time()
                    on_event("enter_still", {"H": round(Hn,3)}, frame.copy())
            else:
                dwell = time.time() - state_since
                if dwell >= STILL_MIN_DUR and abs((dwell%1.0)-0.0) < 1/60:
                    on_event("still_dwell", {"sec": round(dwell,2), "H": round(Hn,3)}, frame.copy())
                if Hn >= STILL_THR_EXIT:
                    on_event("exit_still", {"H": round(Hn,3), "dur": round(dwell,2)}, frame.copy())
                    state="MOTION"; state_since=time.time()

            # ---- Chat (non-blocking) ----
            incoming = chat_io.poll()
            if incoming is not None:
                chat.last = chat.reply(incoming, {"curiosity": curiosity})
                last_chat_t = time.time()

            # ---- HUD ----
            frame_cnt += 1
            now = time.time()
            if now - last_fps_t >= 1.0:
                fps = frame_cnt / (now - last_fps_t)
                last_fps_t = now; frame_cnt = 0

            cv.putText(frame, f"FPS:{fps:4.1f}", (10,22), cv.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)
            cv.putText(frame, f"wV:{w_video:.2f} wA:{w_audio:.2f}  Hn:{Hn:0.2f}", (10,44),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv.putText(frame, f"STATE:{state}  enter<={STILL_THR_ENTER:.2f} exit>={STILL_THR_EXIT:.2f}",
                       (10,64), cv.FONT_HERSHEY_SIMPLEX, 0.5, (180,220,255), 1)

            if time.time()-last_chat_t<5:
                frame = draw_text(frame, f"BOT: {chat.last}", (10, SIZE[1]-20), (255,255,255))

            cv.imshow("Nori AV Shannon++ (Integrated)", frame)
            k = cv.waitKey(1)&0xFF
            if k==27: break
            elif k in (ord('z'), ord('Z')):
                w_audio = float(np.clip(w_audio + 0.05, 0, 1)); w_video = 1 - w_audio
            elif k in (ord('a'), ord('A')):
                w_audio = float(np.clip(w_audio - 0.05, 0, 1)); w_video = 1 - w_audio
            elif k==ord('['):  STILL_THR_ENTER = float(np.clip(STILL_THR_ENTER-0.02, 0.00, STILL_THR_EXIT-0.02))
            elif k==ord(']'):  STILL_THR_ENTER = float(np.clip(STILL_THR_ENTER+0.02, 0.00, STILL_THR_EXIT-0.02))
            elif k==ord('{'):  STILL_THR_EXIT  = float(np.clip(STILL_THR_EXIT -0.02, STILL_THR_ENTER+0.02, 1.00))
            elif k==ord('}'):  STILL_THR_EXIT  = float(np.clip(STILL_THR_EXIT +0.02, STILL_THR_ENTER+0.02, 1.00))
            elif k in (ord('c'), ord('C')):  # quick check latest question
                q=curiosity.latest_question(); print("[WHAT]", q[3] if q else "まだ質問なし")

    finally:
        chat_io.stop()
        cap.release()
        try:
            if audio_stream: audio_stream.stop(); audio_stream.close()
        except Exception: pass
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
