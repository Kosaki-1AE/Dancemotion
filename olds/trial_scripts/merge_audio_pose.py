# merge_audio_pose.py
import os
from collections import OrderedDict

import torch

os.makedirs("experts/merged", exist_ok=True)

def slerp(a, b, t=0.35):
    # 形が同じTensor同士に適用（失敗したら平均にフォールバック）
    eps = 1e-9
    a_f = a.reshape(-1)
    b_f = b.reshape(-1)
    an = a_f.norm() + eps
    bn = b_f.norm() + eps
    a_u = a_f / an; b_u = b_f / bn
    dot = (a_u * b_u).sum().clamp(-1+1e-6, 1-1e-6)
    theta = torch.acos(dot)
    w1 = torch.sin((1-t)*theta) / torch.sin(theta + eps)
    w2 = torch.sin(t*theta) / torch.sin(theta + eps)
    out = (w1 * a_f * an) + (w2 * b_f * bn)
    return out.reshape_as(a)

# ====== ベストckptを指定（必要ならファイル名を変更して）======
ck_audio_path = "experts/audio/ckpt_seed0.pt"
ck_pose_path  = "experts/pose/ckpt_seed0.pt"

ck_audio = torch.load(ck_audio_path, map_location="cpu")
ck_pose  = torch.load(ck_pose_path,  map_location="cpu")

merged = OrderedDict()
for k in ck_audio.keys():
    va = ck_audio[k]; vb = ck_pose.get(k, None)

    if vb is None:
        print(f"skip {k} (not found in pose ckpt)")
        continue

    # shape が一致してるなら普通にマージ
    if va.shape == vb.shape:
        if k.startswith("resp."):
            merged[k] = (va + vb) / 2
        elif k.startswith("fc."):
            try:
                merged[k] = slerp(va, vb, t=0.35)
            except Exception:
                merged[k] = (va + vb) / 2
        else:
            merged[k] = (va + vb) / 2
    else:
        # shape が違う場合は concat する（M2N2っぽい強引融合）
        print(f"⚠ shape mismatch at {k}: {va.shape} vs {vb.shape} → concat")
        merged[k] = torch.cat([va.flatten(), vb.flatten()])

out_path = "experts/merged/ckpt_m2n2.pt"
torch.save(merged, out_path)
print("saved:", out_path)
