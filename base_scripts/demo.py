# responsibility_allow/demo.py　一応全体の計算まとめがここね
import numpy as np
from acts_core import get_activation  # 既存の活性化モジュール
from analyze import analyze_activation
from complex_ops import split_real_imag  # 複素ベクトルの実部/虚部分解
from flow import FlowState, make_ev_decider_core
from linops import linear_transform


def print_pair_result(x, z, pos_name, neg_name, out):
    out_pos = out["detail"]["out_pos"]
    out_neg = out["detail"]["out_neg"]
    out_neg_abs = np.abs(out_neg)
    out_combined = out_pos + out_neg_abs

    print("=" * 72)
    print(f"使用活性（正/負）: {pos_name} / {neg_name}")
    print("入力ベクトル:", x)
    print("線形変換後:", z)
    print("正の責任（愛）:", out_pos)
    print("負の責任（えぐみ）:", out_neg)
    print("負の強さ（|えぐみ|）:", out_neg_abs)
    print("愛とえぐみの合算（愛 + |えぐみ|）:", out_combined)
    print("正負の和（検算・原点復帰）:", out_pos + out_neg)
    print("正負の差（安全地帯の幅）:", out_pos - out_neg)
    print()
    print("なお愛とえぐみの比較がこちら:")
    for i, (pos, negs) in enumerate(zip(out_pos, out_neg_abs)):
        if pos > negs: result = "愛が優勢"
        elif pos < negs: result = "エグみが優勢"
        else: result = "拮抗"
        print(f"成分{i}: 正={pos:.2f}, 負(強さ)={negs:.2f} → {result}")
    print()
    print("よってダンサーがどういう配分に見えるか（感覚派の“流れ”の数値化）:", out_combined)
    print()

if __name__ == "__main__":
    # 入力とパラメタ
    x = np.array([1.0, -2.0, 3.0])
    W = np.array([[0.5, -1.0, 0.3],
                  [0.8,  0.2, -0.5],
                  [-0.6, 0.4,  1.0]])
    b = np.array([0.1, -0.2, 0.3])

    z = linear_transform(x, W, b)

    pos_names = ["relu", "leaky_relu", "sigmoid", "tanh", "silu", "gelu"]
    neg_names = [f"neg_{n}" for n in pos_names]

    for pos_name, neg_name in zip(pos_names, neg_names):
        pos_fn = get_activation(pos_name)
        neg_fn = get_activation(neg_name)
        out = analyze_activation(
            x, W, b, pos_fn, neg_fn,
            tau=1.0, topk=None, fluct_mode="none", center="auto",
            name_pos=pos_name, name_neg=neg_name
        )
        # will_event 風にまとめ直す（出力フォーマット合わせ）
        will_like = {
            "commit": bool(out["p_pos"] >= 0.5),
            "p_hat": float(out["p_pos"]),
            "theta": 0.5,
            "polarity": 1 if out["pos_sum"] >= out["neg_sum"] else -1,
            "intensity": abs(out["delta"]),
            "detail": out
        }
        print_pair_result(x, z, pos_name, neg_name, will_like)

    # Flow デモ（おまけ）
    pos_fn = get_activation("silu")
    neg_fn = get_activation("neg_silu")
    decider = make_ev_decider_core(x, W, b, pos_fn=pos_fn, neg_fn=neg_fn,
                                   tau=1.0, center="auto",
                                   fluct_mode="logit_gauss", fluct_kwargs={"sigma":0.4, "seed":123},
                                   theta_init=0.60)
    flow = FlowState(dim=256, theta0=0.60, ema=0.05)
    msgs = [
        "目的：Stillness×矢×ノリエントロピーの相互作用を検証する。",
        "仮説：Stillness↑で|Δ|↑、整合↑でcommit率↑。",
        "反例：場の方向を急反転させてもcommitは維持されるか？",
        "設計：条件を変化させ t=60 までログ収集。",
        "実験：2×2 条件で commit 率と |Δ| を算出。",
        "評価：単調性/相関/反例を要約し仮説を判定。",
        "決定：v0モデル採用。次は層間相互情報を追加。",
    ]
    last = 0.0
    for t, m in enumerate(msgs):
        out = flow.step(m, base_delta=last, decider_fn=decider)
        last = 0.2 * last + 0.1 * (1 if out["commit"] else -0.2)
        print(f"[Flow] t={t:02d} commit={out['commit']} p_hat={out['p_hat']:.3f} "
              f"theta={out['theta_now']:.2f} d_hat={np.round(out['d_hat'],3)} msg='{m[:24]}{'…' if len(m)>24 else ''}'")
