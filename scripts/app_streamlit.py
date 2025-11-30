# app_streamlit.py
import numpy as np
import streamlit as st
import torch

from responsibility_layer import ResponsibilityLayer

st.set_page_config(page_title="Responsibility Live", layout="wide")

st.title("Responsibility Layer Live Monitor")
in_dim = st.sidebar.number_input("in_dim", 1, 256, 16)
out_dim = st.sidebar.number_input("out_dim", 1, 256, 16)
seed = st.sidebar.number_input("seed", 0, 9999, 42)

rng = np.random.default_rng(seed)
x = rng.normal(size=(200, in_dim)).astype(np.float32)  # 200タイムステップ想定

model = ResponsibilityLayer(in_dim, out_dim)
with torch.no_grad():
    out = model(torch.tensor(x))
    pos = out["pos"].numpy().mean(axis=1)          # 成分平均でざっくり
    neg = out["neg"].numpy().mean(axis=1)
    S = out["safety_width"].numpy().mean(axis=1)
    C = out["combined"].numpy().mean(axis=1)

st.line_chart({"pos": pos, "neg": neg, "safety_width": S, "combined": C})
st.caption("pos: 愛 / neg: えぐみ / safety_width: 余白 / combined: 統合責任")
