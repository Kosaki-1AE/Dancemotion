from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

Node = str
Array = np.ndarray


# -----------------------------
# util
# -----------------------------
def softmax(x: Array, temp: float = 1.0) -> Array:
    z = x / max(temp, 1e-9)
    z = z - np.max(z)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)

def clamp_int(x: Array, lo: int, hi: Array | int) -> Array:
    return np.minimum(np.maximum(x, lo), hi).astype(int)


# -----------------------------
# graph (場の結合)
# -----------------------------
@dataclass
class Graph:
    nodes: List[Node]
    edges: List[Tuple[Node, Node, float]]  # src -> dst

    def idx(self) -> Dict[Node, int]:
        return {n: i for i, n in enumerate(self.nodes)}

    def W(self) -> Array:
        n = len(self.nodes)
        mat = np.zeros((n, n), dtype=float)
        m = self.idx()
        for a, b, w in self.edges:
            mat[m[a], m[b]] += float(w)
        return mat


# -----------------------------
# gates / mapping
# -----------------------------
@dataclass
class Gates:
    TREE3: Array  # bool: 通行許可(上限解放 + 変換可能性)

@dataclass
class TauPredMap:
    tau_by_state: List[float]
    def max_state(self) -> int:
        return len(self.tau_by_state) - 1
    def tau_pred(self, state: Array) -> Array:
        s = np.clip(state.astype(int), 0, self.max_state())
        return np.array([self.tau_by_state[int(x)] for x in s], dtype=float)


# -----------------------------
# external world G(t)
# -----------------------------
@dataclass
class ExternalG:
    """
    G(t) = 外界から提示される「それしかない」制約
    forced_state[t, i] で、各ノードに強制される状態レベルを与える（希望がないケース含む）
    """
    forced_state: Array  # shape (T, N), int

    def at(self, t: int) -> Array:
        return self.forced_state[t].astype(int)


# -----------------------------
# config
# -----------------------------
@dataclass
class Config:
    # state更新
    inertia_state: float = 0.90
    coupling: float = 0.25

    # D配分（摩擦＋無意識を観測して配分）
    temp: float = 1.0
    entropy_reg: float = 0.10

    # method探索（抗い）
    lr_method: float = 0.25
    method_decay: float = 0.02

    # 摩擦→方法探索→state変換
    alpha_friction: float = 1.0
    beta_method_to_state: float = 0.60

    # 確率的無意識（摩擦の観測ノイズ）
    sigma0: float = 0.02
    sigma_f: float = 0.10

    # TREE3: 通行許可（上限解放）
    blocked_max_offset: int = 1
    tree3_convert_gain: float = 1.0
    blocked_convert_gain: float = 0.2

    # 量子化
    step_size: float = 1.0

    # 再現性
    seed: int = 0

    # -----------------------------
    # Responsibility (蓄積 & 放出)
    # -----------------------------
    resp_gain_friction: float = 1.0   # 摩擦→Responsibility
    resp_gain_cost: float = 1.0       # コスト→Responsibility
    resp_release_rate: float = 0.15   # 「全トリガー」= 毎step放出割合（0.05〜0.3が目安）

    # 放出をどう世界に変換するか（world_freedom）
    freedom_gain_scale: float = 1.0   # 放出→world_freedom 変換係数

    # world_freedom の使い道：G強制の緩和
    g_relax_rate: float = 0.03        # world_freedom が強制圧を削る強さ（小さめ推奨）

    # world_freedom の使い道：TREE3開放
    tree3_open_threshold: float = 6.0 # これだけ貯まったら1ノード開放
    tree3_open_cost: float = 6.0      # 開放に消費するworld_freedom（thresholdと同じでOK）


# -----------------------------
# main field: state + method + D allocation + Responsibility
# -----------------------------
@dataclass
class Field:
    graph: Graph
    gates: Gates
    tau_map: TauPredMap

    # dynamics variables
    state: Array          # (N,) int
    method: Array         # (N,) float

    # NEW: Responsibility + world_freedom
    responsibility: Array # (N,) float
    world_freedom: float = 0.0

    # previous (for logs)
    prev_state: Optional[Array] = None
    prev_method: Optional[Array] = None

    def __post_init__(self) -> None:
        if self.prev_state is None:
            self.prev_state = self.state.copy()
        if self.prev_method is None:
            self.prev_method = self.method.copy()

    def neighbor_mean(self) -> Array:
        W = self.graph.W()
        in_w = W.sum(axis=0) + 1e-9
        return (W.T @ self.state) / in_w

    def unconscious_noise(self, friction: Array, cfg: Config, rng: np.random.Generator) -> Array:
        """
        確率的無意識：摩擦が大きいほど観測が荒れる（把握しきれない）
        """
        sigma = cfg.sigma0 + cfg.sigma_f * np.sqrt(np.maximum(friction, 0.0))
        return rng.normal(0.0, sigma, size=friction.shape)

    def d_allocation(self, observed_friction: Array, cfg: Config) -> Array:
        """
        D配分：観測した摩擦（＋無意識ノイズ）に基づいて資源配分
        """
        base = np.maximum(observed_friction, 0.0) + 1e-8
        p = softmax(base, temp=cfg.temp)

        if cfg.entropy_reg > 0:
            uni = np.ones_like(p) / len(p)
            p = (1 - cfg.entropy_reg) * p + cfg.entropy_reg * uni
        return p

    def _apply_world_freedom_to_forced(self, g_forced: Array, cfg: Config) -> Array:
        """
        world_freedom による外界強制の緩和：
        「それしかない」が少しずつ柔らかくなる
        """
        if self.world_freedom <= 0:
            return g_forced

        relaxed = g_forced.astype(float) - cfg.g_relax_rate * self.world_freedom
        relaxed = np.clip(relaxed, 0.0, float(self.tau_map.max_state()))
        return np.rint(relaxed).astype(int)

    def _maybe_open_tree3(self, cfg: Config) -> List[Node]:
        """
        world_freedom が溜まったら TREE3 を開く（通行許可が増える）
        開放対象は「まだ閉じているノード」の中から、摩擦源になりやすい順で開けるのが自然だが
        ここでは単純に先頭から開ける（必要なら優先順位ロジック作る）
        """
        opened: List[Node] = []
        # 何回でも開ける（十分溜まってれば複数開放）
        while self.world_freedom >= cfg.tree3_open_threshold:
            closed_idx = np.where(~self.gates.TREE3)[0]
            if len(closed_idx) == 0:
                break
            i = int(closed_idx[0])
            self.gates.TREE3[i] = True
            self.world_freedom -= cfg.tree3_open_cost
            opened.append(self.graph.nodes[i])
        return opened

    def step(self, *, g_forced: Array, cfg: Config, rng: np.random.Generator) -> Dict[str, Dict]:
        """
        1 step:
          0) Responsibility放出→world_freedom→G緩和/TREE3開放（「全部トリガー」なので毎step）
          1) Gの強制状態とのズレ→摩擦
          2) 摩擦を観測（確率的無意識）してDが配分 p を作る
          3) 抗い：method を更新（探索コスト発生）
          4) method改善が state に変換される（TREE3が通行許可）
          5) 場の結合（neighbor）＋慣性で state 更新
          6) Responsibility蓄積→次stepで放出…
        """
        N = len(self.graph.nodes)
        global_max = self.tau_map.max_state()
        blocked_max = max(0, global_max - cfg.blocked_max_offset)
        max_per_node = np.where(self.gates.TREE3, global_max, blocked_max)

        # ------------------------------------------------------------
        # 0) Responsibility release (ALL triggers = every step)
        #    - 放出された分が world_freedom になり、世界が柔らかくなる
        # ------------------------------------------------------------
        released = cfg.resp_release_rate * self.responsibility
        self.responsibility = np.maximum(self.responsibility - released, 0.0)

        self.world_freedom += cfg.freedom_gain_scale * float(np.sum(released))

        # world_freedom を使って TREE3を開く（通行許可増加）
        opened_nodes = self._maybe_open_tree3(cfg)

        # world_freedom を使って G強制を緩和
        g_forced_eff = self._apply_world_freedom_to_forced(g_forced, cfg)

        # ------------------------------------------------------------
        # 1) friction: 強制(外界)と内部(state)のズレ
        # ------------------------------------------------------------
        friction = cfg.alpha_friction * np.abs(g_forced_eff.astype(float) - self.state.astype(float))

        # ------------------------------------------------------------
        # 2) probabilistic unconscious: 摩擦の観測ノイズ
        # ------------------------------------------------------------
        noise = self.unconscious_noise(friction, cfg, rng)
        observed = np.clip(friction + noise, 0.0, None)

        # ------------------------------------------------------------
        # 3) D allocation: 観測摩擦に基づく配分
        # ------------------------------------------------------------
        p = self.d_allocation(observed, cfg)

        # ------------------------------------------------------------
        # 4) method search (抗い)
        # ------------------------------------------------------------
        method_update = cfg.lr_method * (p * observed) - cfg.method_decay * self.method
        method_new = self.method + method_update

        cost = np.abs(method_new - self.method)

        # ------------------------------------------------------------
        # 5) method -> state conversion (TREE3が通行許可)
        # ------------------------------------------------------------
        convert_gain = np.where(self.gates.TREE3, cfg.tree3_convert_gain, cfg.blocked_convert_gain)
        method_effect = cfg.beta_method_to_state * convert_gain * (method_new - self.method)

        # ------------------------------------------------------------
        # 6) state dynamics with field coupling
        # ------------------------------------------------------------
        s = self.state.astype(float)
        neigh = self.neighbor_mean()

        s_cont = (
            cfg.inertia_state * s
            + cfg.coupling * (neigh - s)
            + method_effect
        )
        s_cont = s + cfg.step_size * (s_cont - s)

        state_new = np.rint(s_cont).astype(int)
        state_new = clamp_int(state_new, 0, max_per_node)

        # ------------------------------------------------------------
        # 7) Responsibility accumulate (ALL components)
        #    「全部」= 摩擦 + コスト + 選ばされ体験(= friction自体) を全部入れる
        # ------------------------------------------------------------
        resp_add = cfg.resp_gain_friction * friction + cfg.resp_gain_cost * cost
        self.responsibility += resp_add

        # update prev
        self.prev_state = self.state.copy()
        self.prev_method = self.method.copy()
        self.state = state_new
        self.method = method_new

        # logs
        return {
            "state": {self.graph.nodes[i]: int(state_new[i]) for i in range(N)},
            "tau_pred": {self.graph.nodes[i]: float(self.tau_map.tau_pred(state_new)[i]) for i in range(N)},
            "forced": {self.graph.nodes[i]: int(g_forced[i]) for i in range(N)},
            "forced_eff": {self.graph.nodes[i]: int(g_forced_eff[i]) for i in range(N)},
            "friction": {self.graph.nodes[i]: float(friction[i]) for i in range(N)},
            "observed": {self.graph.nodes[i]: float(observed[i]) for i in range(N)},
            "p_alloc": {self.graph.nodes[i]: float(p[i]) for i in range(N)},
            "cost": {self.graph.nodes[i]: float(cost[i]) for i in range(N)},
            "responsibility": {self.graph.nodes[i]: float(self.responsibility[i]) for i in range(N)},
            "released_sum": float(np.sum(released)),
            "world_freedom": float(self.world_freedom),
            "opened_TREE3": opened_nodes,
        }


# -----------------------------
# demo
# -----------------------------
if __name__ == "__main__":
    nodes = ["G", "B", "M", "C1", "C2", "S", "Q"]
    edges = [
        ("G", "B", 1.0),
        ("B", "M", 1.0),
        ("M", "C1", 1.0),
        ("C1", "C2", 1.0),
        ("G", "S", 1.0),
        ("S", "Q", 1.0),
        ("Q", "C2", 1.0),
    ]
    graph = Graph(nodes=nodes, edges=edges)

    # 最初はC2だけTREE3解放（通行許可）
    TREE3 = np.array([False, False, False, False, True, False, False], dtype=bool)
    gates = Gates(TREE3=TREE3)

    tau_map = TauPredMap([0.7, 1.0, 1.3, 1.7, 2.2])

    state0 = np.array([0, 1, 2, 3, 3, 1, 2], dtype=int)
    method0 = np.zeros_like(state0, dtype=float)
    resp0 = np.zeros_like(state0, dtype=float)

    cfg = Config(seed=42)
    rng = np.random.default_rng(cfg.seed)

    # 外界G(t): 強制圧
    T = 12
    forced = np.tile(state0, (T, 1))
    idx = graph.idx()
    forced[:, idx["G"]] = 3
    forced[:, idx["B"]] = 3
    forced[:, idx["S"]] = 2
    ext = ExternalG(forced_state=forced)

    field = Field(
        graph=graph,
        gates=gates,
        tau_map=tau_map,
        state=state0,
        method=method0,
        responsibility=resp0,
        world_freedom=0.0
    )

    for t in range(T):
        logs = field.step(g_forced=ext.at(t), cfg=cfg, rng=rng)
        print(f"\n--- t={t} ---")
        print("state       :", logs["state"])
        print("tau^pred    :", logs["tau_pred"])
        print("forced      :", logs["forced"])
        print("forced_eff  :", logs["forced_eff"], " (world_freedomで緩和後)")
        print("friction    :", logs["friction"])
        print("p(D)        :", logs["p_alloc"])
        print("cost        :", logs["cost"])
        print("R           :", {k: round(v, 3) for k, v in logs["responsibility"].items()})
        print("released_sum:", round(logs["released_sum"], 4), " world_freedom:", round(logs["world_freedom"], 4))
        if logs["opened_TREE3"]:
            print("TREE3 opened:", logs["opened_TREE3"])
