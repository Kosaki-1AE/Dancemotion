# -*- coding: utf-8 -*-
"""
Beat Quantum x librosa x MediaPipe — 自動カメラ版
起動するとデフォルトカメラ(0)を使い、20秒だけ動きエネルギを測定して量子リズム化。
クリックWAVも自動で保存。
"""

import math, sys, time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import Aer
from qiskit.compiler import transpile

try:
    import cv2
    import mediapipe as mp
    MP_OK = True
except Exception:
    MP_OK = False

try:
    import librosa
    LB_OK = True
except Exception:
    LB_OK = False

# ========== 拍量子パラメータ ==========
@dataclass
class BeatParams:
    tempo_bpm: float = 120.0
    bars: int = 4
    beats_per_bar: int = 4
    base_hit_prob: float = 0.8
    stillness_prob: float = 0.0
    humanize_prob_jitter: float = 0.05
    responsibility_phase_ms: float = 0.0
    humanize_phase_ms: float = 5.0
    global_phase_rad: float = 0.0
    seed: int = 42

def beat_duration_sec(bpm: float) -> float: return 60.0 / bpm
def clamp01(x: float) -> float: return float(np.clip(x, 0.0, 1.0))
def prob_to_ry_theta(p_hit: float) -> float:
    return 2.0 * math.asin(math.sqrt(np.clip(p_hit,0.0,1.0)))
def ms_to_rad(ms: float, bpm: float) -> float:
    beat_ms = beat_duration_sec(bpm)*1000.0
    return 2.0*math.pi*(ms/beat_ms)
def rad_to_ms(phi_rad: float, bpm: float) -> float:
    beat_ms = beat_duration_sec(bpm)*1000.0
    return (phi_rad/(2.0*math.pi))*beat_ms

# ========== 量子回路 ==========
def build_rhythm_circuit(params: BeatParams) -> Tuple[QuantumCircuit, List[float], List[float]]:
    rng = np.random.default_rng(params.seed)
    n_beats = params.bars*params.beats_per_bar
    q = QuantumRegister(1,"q"); c = ClassicalRegister(n_beats,"c")
    qc = QuantumCircuit(q,c,name="BeatQuantumRhythm")
    thetas, phis = [], []
    for i in range(n_beats):
        p = clamp01(params.base_hit_prob - params.stillness_prob + rng.normal(0,params.humanize_prob_jitter))
        theta = prob_to_ry_theta(p); qc.ry(theta,q[0])
        phase_ms = params.responsibility_phase_ms + rng.normal(0,params.humanize_phase_ms)
        phi = ms_to_rad(phase_ms,params.tempo_bpm)+params.global_phase_rad
        qc.rz(phi,q[0]); qc.measure(q[0],c[i]); qc.reset(q[0])
        thetas.append(theta); phis.append(phi)
    return qc, thetas, phis

def sample_rhythm(qc: QuantumCircuit) -> str:
    backend = Aer.get_backend("aer_simulator")
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=1).result()
    bitstr = list(result.get_counts().keys())[0][::-1]  # 拍順に反転
    return bitstr

def make_timetable(bitstr: str, params: BeatParams, per_beat_phase_ms: List[float]) -> List[dict]:
    beat_sec = beat_duration_sec(params.tempo_bpm)
    return [{
        "index":i,
        "beat_time_sec":i*beat_sec,
        "phase_ms":per_beat_phase_ms[i],
        "event_time_sec":max(0.0, i*beat_sec + per_beat_phase_ms[i]/1000.0),
        "event":"HIT" if b=="1" else "REST"
    } for i,b in enumerate(bitstr)]

# ========== WAV書き出し ==========
def render_click_wav(table: List[dict], bpm: float, out="beat_quantum.wav"):
    import wave
    sr=44100; beat_sec=60.0/bpm
    total=table[-1]["beat_time_sec"]+beat_sec+0.5
    buf=np.zeros(int(sr*total),dtype=np.float32)
    t=np.arange(int(sr*0.03))/sr; clk=np.sin(2*np.pi*880*t).astype(np.float32)
    for r in table:
        if r["event"]!="HIT": continue
        idx=int(sr*r["event_time_sec"]); buf[idx:idx+len(clk)]+=clk
    if np.max(np.abs(buf))>0: buf=buf/np.max(np.abs(buf))*0.9
    with wave.open(out,"wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((buf*32767).astype(np.int16).tobytes())
    print(f"[ok] saved {out}")

# ========== MediaPipe: 動き解析 ==========
def motion_energy_from_camera(cam_id:int=0, tempo_bpm:float=120.0, limit_sec:float=20.0):
    if not MP_OK:
        print("[warn] mediapipe未導入")
        return [],0.0
    cap=cv2.VideoCapture(cam_id)
    mp_pose=mp.solutions.pose; pose=mp_pose.Pose()
    sr=int(cap.get(cv2.CAP_PROP_FPS)) or 30; beat_sec=beat_duration_sec(tempo_bpm)
    energies=[]; prev=None; start=time.time()
    while True:
        ret,frame=cap.read(); 
        if not ret: break
        if (time.time()-start)>limit_sec: break
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=pose.process(frame)
        if res.pose_landmarks is None: energies.append(0.0); continue
        pts=np.array([[p.x,p.y,p.z] for p in res.pose_landmarks.landmark])
        if prev is None: prev=pts; energies.append(0.0); continue
        e=float(np.sqrt(((pts-prev)**2).sum(axis=1)).mean())
        energies.append(e); prev=pts
    cap.release(); pose.close()
    if len(energies)==0: return [],0.0
    dur=len(energies)/sr; n_beats=int(np.floor(dur/beat_sec))
    per_beat=[float(np.mean(energies[int(i*beat_sec*sr):int((i+1)*beat_sec*sr)])) for i in range(n_beats)]
    return per_beat,0.0

# ========== main ==========
if __name__=="__main__":
    print("[info] Starting auto-camera Beat Quantum...")
    base=BeatParams()
    per_beat_motion,phase_ms=motion_energy_from_camera(cam_id=0,tempo_bpm=base.tempo_bpm,limit_sec=20.0)
    if len(per_beat_motion)>0:
        m=np.clip((np.array(per_beat_motion)-0.0)/0.02,0.0,1.0)
        base.base_hit_prob=0.5+0.5*float(np.mean(m))
        base.stillness_prob=0.4*(1.0-np.mean(m))
    qc,thetas,phis=build_rhythm_circuit(base)
    per_beat_phase_ms=[rad_to_ms(phi,base.tempo_bpm) for phi in phis]
    bitstr=sample_rhythm(qc)
    table=make_timetable(bitstr,base,per_beat_phase_ms)
    print("=== Beat Quantum Rhythm (AutoCamera) ===")
    for r in table:
        print(f"beat{r['index']:02d}: {r['event']} @ {r['event_time_sec']:.3f}s (phase={r['phase_ms']:+.1f}ms)")
    render_click_wav(table, base.tempo_bpm)
