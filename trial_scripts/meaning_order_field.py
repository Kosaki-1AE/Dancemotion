# meaning_order_field.py
# Reference implementation of "Meaning-Order Vector Field"
# Author: ChatGPT (with user spec)
# -------------------------------------------------------------
# M(x,y) = α R(x,y) + β ∇I(x,y) − δ ∇H(x,y)
# R: responsibility vector field from agents (pos, direction, weight)
# I: interference scalar field from two sources (double-slit-like)
# H: Shannon entropy from softmax of agent influence
#
# Usage:
#   python meaning_order_field.py
# Outputs:
#   meaning_order_field_I_M.(png|svg)
#   meaning_order_field_H_M.(png|svg)
#   responsibility_field_only.(png|svg)

import numpy as np
import matplotlib.pyplot as plt

def build_field(
    L=8.0, N=100,
    agents=None,
    sources=None,
    wavelength=3.0,
    alpha=1.2, beta=0.9, delta=0.8,
    influence_sigma=2.3,
    quiver_step=4
):
    # Grid
    xs = np.linspace(-L, L, N)
    ys = np.linspace(-L, L, N)
    X, Y = np.meshgrid(xs, ys)

    # Default agents
    if agents is None:
        agents = [
            {"pos": np.array([-3.0, -1.0]), "dir": np.array([ 1.0,  0.2]), "w": 1.0},
            {"pos": np.array([ 2.5, -2.0]), "dir": np.array([-0.6,  0.8]), "w": 0.9},
            {"pos": np.array([-1.0,  2.5]), "dir": np.array([ 0.3, -0.9]), "w": 0.8},
        ]

    # Default double-slit-like sources
    if sources is None:
        sources = [np.array([0.0,  0.5]), np.array([0.0, -0.5])]

    # 1) Responsibility field R
    Rx = np.zeros_like(X); Ry = np.zeros_like(Y)
    for ag in agents:
        d = ag["dir"] / (np.linalg.norm(ag["dir"]) + 1e-12)
        dx = X - ag["pos"][0]; dy = Y - ag["pos"][1]
        dist2 = dx*dx + dy*dy
        infl = ag["w"] * np.exp(-dist2/(2*influence_sigma**2))
        Rx += infl * d[0]; Ry += infl * d[1]

    # 2) Interference field I
    k = 2*np.pi / wavelength
    def rdist(P):
        return np.sqrt((X-P[0])**2 + (Y-P[1])**2)
    r1 = rdist(sources[0]); r2 = rdist(sources[1])
    I = np.cos(k*r1) + np.cos(k*r2)
    dIx, dIy = np.gradient(I, xs, ys, edge_order=2)

    # 3) Entropy field H (softmax over agent influences)
    raw = []
    for ag in agents:
        dx = X - ag["pos"][0]; dy = Y - ag["pos"][1]
        dist2 = dx*dx + dy*dy
        raw.append(ag["w"] * np.exp(-dist2/(2*influence_sigma**2)))
    raw = np.stack(raw, axis=0)
    raw_shift = raw - np.max(raw, axis=0, keepdims=True)
    p = np.exp(raw_shift); p = p / (np.sum(p, axis=0, keepdims=True) + 1e-12)
    H = -np.sum(p * np.log(p + 1e-12), axis=0)
    dHx, dHy = np.gradient(H, xs, ys, edge_order=2)

    # 4) Meaning-Order field M
    Mx = alpha*Rx + beta*dIx - delta*dHx
    My = alpha*Ry + beta*dIy - delta*dHy

    # Quiver sampling
    Xq, Yq = X[::quiver_step, ::quiver_step], Y[::quiver_step, ::quiver_step]
    Mxq, Myq = Mx[::quiver_step, ::quiver_step], My[::quiver_step, ::quiver_step]

    return {
        "X": X, "Y": Y, "xs": xs, "ys": ys,
        "Rx": Rx, "Ry": Ry,
        "I": I, "dIx": dIx, "dIy": dIy,
        "H": H, "dHx": dHx, "dHy": dHy,
        "Mx": Mx, "My": My,
        "Xq": Xq, "Yq": Yq, "Mxq": Mxq, "Myq": Myq,
        "agents": agents, "sources": sources
    }

def plot_and_save(fields):
    X = fields["X"]; Y = fields["Y"]
    xs = fields["xs"]; ys = fields["ys"]
    Rx, Ry = fields["Rx"], fields["Ry"]
    I = fields["I"]; H = fields["H"]
    Mx, My = fields["Mx"], fields["My"]
    Xq, Yq = fields["Xq"], fields["Yq"]
    Mxq, Myq = fields["Mxq"], fields["Myq"]
    agents = fields["agents"]; sources = fields["sources"]

    # Fig 1: I + M
    plt.figure(figsize=(8,6))
    plt.imshow(I, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin='lower', aspect='equal')
    plt.quiver(Xq, Yq, Mxq, Myq, scale=60)
    for ag in agents:
        plt.plot(ag["pos"][0], ag["pos"][1], 'o')
        d = ag["dir"]/np.linalg.norm(ag["dir"])
        plt.arrow(ag["pos"][0], ag["pos"][1], 0.8*d[0], 0.8*d[1],
                  head_width=0.25, length_includes_head=True)
    for s in sources:
        plt.plot(s[0], s[1], 's')
    plt.title("Interference I(x,y) with Meaning-Order Vector Field M(x,y)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    plt.savefig("meaning_order_field_I_M.png", dpi=180)
    plt.savefig("meaning_order_field_I_M.svg")
    plt.close()

    # Fig 2: H + M
    plt.figure(figsize=(8,6))
    plt.imshow(H, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin='lower', aspect='equal')
    cs = plt.contour(X, Y, H, levels=10, linewidths=0.8)
    plt.clabel(cs, inline=True, fontsize=8)
    plt.quiver(Xq, Yq, Mxq, Myq, scale=60)
    for ag in agents:
        plt.plot(ag["pos"][0], ag["pos"][1], 'o')
    plt.title("Entropy H(x,y) with Meaning-Order Vector Field M(x,y)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    plt.savefig("meaning_order_field_H_M.png", dpi=180)
    plt.savefig("meaning_order_field_H_M.svg")
    plt.close()

    # Fig 3: R only
    plt.figure(figsize=(8,6))
    plt.quiver(Xq, Yq, Rx[::4, ::4], Ry[::4, ::4], scale=60)
    for ag in agents:
        plt.plot(ag["pos"][0], ag["pos"][1], 'o')
        d = ag["dir"]/np.linalg.norm(ag["dir"])
        plt.arrow(ag["pos"][0], ag["pos"][1], 0.8*d[0], 0.8*d[1],
                  head_width=0.25, length_includes_head=True)
    plt.title("Responsibility Vector Field R(x,y)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    plt.savefig("responsibility_field_only.png", dpi=180)
    plt.savefig("responsibility_field_only.svg")
    plt.close()

if __name__ == "__main__":
    fields = build_field()
    plot_and_save(fields)
    print("Saved: meaning_order_field_I_M.(png|svg), meaning_order_field_H_M.(png|svg), responsibility_field_only.(png|svg)")