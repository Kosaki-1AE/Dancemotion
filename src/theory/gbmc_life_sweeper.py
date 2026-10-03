# gbmc_life_sweeper_v2.py
# KQI Theory Implementation: GBMC Integrated Model
# Buffer (B) = LifeGame (Time/Tau) + Minesweeper (Space/Sigma)

import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np

# -------------------------
# Config
# -------------------------
H, W = 20, 40
MINES = 100            # 地雷数（sigmaの総量）
TICK_SEC = 0.1
SEED = None
AUTO_OPEN_EVERY = 0    # 0で無効化。>0で自動探索

# KQI Parameters
THETA = 2.0            # Motion閾値 (これを超えないと描画されない)
K_MODE = 'inf'         # Coherenceモード: 0, 1, 'inf'

# -------------------------
# Utils
# -------------------------
def clear():
    os.system("cls" if os.name == "nt" else "clear")

def inb(i, j):
    return 0 <= i < H and 0 <= j < W

def neighbors8(i, j):
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0: continue
            ni, nj = i + di, j + dj
            if inb(ni, nj):
                yield ni, nj

# -------------------------
# Minesweeper Layer (Static Sigma)
# -------------------------
def make_mines():
    coords = [(i, j) for i in range(H) for j in range(W)]
    mines = set(random.sample(coords, MINES))
    field = np.zeros((H, W), dtype=int)
    
    # -1: Mine, 0-8: Adjacent count
    for (i, j) in mines:
        field[i, j] = -1
        
    for i in range(H):
        for j in range(W):
            if field[i, j] == -1: continue
            c = 0
            for ni, nj in neighbors8(i, j):
                if (ni, nj) in mines: c += 1
            field[i, j] = c
    return mines, field

# -------------------------
# Life Layer (Dynamic Tau)
# -------------------------
def life_step(grid: np.ndarray) -> np.ndarray:
    new = grid.copy()
    for i in range(H):
        for j in range(W):
            n = 0
            for ni, nj in neighbors8(i, j):
                n += grid[ni, nj]
            if grid[i, j] == 1 and (n < 2 or n > 3):
                new[i, j] = 0
            elif grid[i, j] == 0 and n == 3:
                new[i, j] = 1
    return new

# -------------------------
# KQI: GBMC Core Logic
# -------------------------
@dataclass
class GBMC_Result:
    G: float
    Q: float
    M: float
    C: float
    D: float  # Final Depth (Visual Output)

def calculate_gbmc(prev_grid, grid, mine_field, i, j) -> GBMC_Result:
    """
    KQI理論に基づく厳密な計算処理
    B = {Sigma(Mine), Tau(Life)}
    """
    # --- 1. Genesys (G) = r * omega ---
    # r (Range): LifeGameにおける近傍の生命密度 (0~8)
    life_neighbors = 0
    same_state_neighbors = 0
    for ni, nj in neighbors8(i, j):
        life_neighbors += grid[ni, nj]
        if grid[ni, nj] == grid[i, j]:
            same_state_neighbors += 1
    
    r = float(life_neighbors)
    
    # omega (Angular Velocity): 状態変化の有無 (変化=1.0, 静止=0.1)
    # ※完全な0にすると計算が消滅するので、静止エネルギー0.1を残す
    did_change = (prev_grid[i, j] != grid[i, j])
    omega = 2.0 if did_change else 0.5
    
    G = r * omega

    # --- 2. Base (B) = {sigma, tau} ---
    # sigma (Resistance): マインスイーパの危険度により定義
    # 地雷(-1)は最大のノイズ。数字が大きいほどノイズが高いとするか、
    # 逆に「数字が大きい＝情報量が多い」とするかだが、
    # ここでは「地雷原の圧」として、数字が大きいほど sigma を大きく（Qを抑制）してみる
    mine_val = mine_field[i, j]
    if mine_val == -1:
        sigma = 10.0 # 地雷そのものは最大の不安定要素
    else:
        # 数字が0(安全)ならsigma小、数字が大きいほどsigma大
        sigma = 1.0 + (mine_val * 0.5)

    # tau (Time/Density): 生命の密度
    tau = life_neighbors / 8.0

    # --- 3. Quality (Q) = G^2 / sigma ---
    # エネルギー生成
    Q = (G ** 2) / sigma

    # --- 4. Motion (M) = H(Q - theta) ---
    # 閾値判定
    M = 1.0 if Q >= THETA else 0.0

    # --- 5. Coherence (C) ---
    # k='inf' モード: 指数関数的共鳴
    # 周囲との同期率(same_state)をタウとして扱う
    coherence_base = same_state_neighbors / 8.0
    if K_MODE == 'inf':
        C = math.exp(coherence_base) # e^x
    elif K_MODE == 1:
        C = 1.0 + coherence_base
    else:
        C = 1.0

    # --- 6. Depth (D) = Q * C * M ---
    # Motionゲートが閉じている(M=0)なら観測されない
    D = Q * C * M

    return GBMC_Result(G, Q, M, C, D)

def score_to_char(d: float) -> str:
    # Depth値をASCIIにマッピング
    # Dは 0 ~ 50 くらいまで跳ね上がる可能性があるため対数スケール等で調整
    if d <= 0: return " "
    
    # マッピング用ランプ（薄い→濃い）
    ramp = " .`^:;~+*?#%@"
    
    # スケーリング (適宜調整)
    val = math.sqrt(d) * 1.5
    idx = int(val)
    idx = max(0, min(len(ramp) - 1, idx))
    return ramp[idx]

# -------------------------
# Gameplay & Render
# -------------------------
def open_cell(visible, mines, field, r, c):
    if not inb(r, c): return False, "範囲外"
    if visible[r, c]: return False, "既開示"
    visible[r, c] = True
    if (r, c) in mines: return True, "BOOM"
    
    # 0の場合は連鎖開放（BFS）
    if field[r, c] == 0:
        q = [(r, c)]
        visited = set([(r, c)])
        while q:
            curr_r, curr_c = q.pop(0)
            for ni, nj in neighbors8(curr_r, curr_c):
                if inb(ni, nj) and not visible[ni, nj]:
                    visible[ni, nj] = True
                    if field[ni, nj] == 0 and (ni, nj) not in visited:
                        visited.add((ni, nj))
                        q.append((ni, nj))
    return False, "OK"

def pick_auto_open_candidate(visible, field):
    # 完全オートではなく「開いてない場所」をランダムに選ぶ（プレイヤーの代行）
    # ズルなし。
    cand = []
    for i in range(H):
        for j in range(W):
            if not visible[i, j]:
                cand.append((i, j))
    if not cand: return None
    return random.choice(cand)

def render(prev, grid, visible, mines, field, tick, last_msg=""):
    clear()
    
    # 統計計算
    total_q = 0.0
    active_m = 0
    
    # 画面バッファ生成
    lines = []
    
    # フレームごとのD値計算
    # ※表示用ループ内で計算するが、本来はMatrix計算で一括処理すべき箇所
    # 今回は可読性と移植性重視でループ処理
    for i in range(H):
        row_chars = []
        for j in range(W):
            if visible[i, j]:
                # 開示済みセル：Minesweeperの現実を表示
                if (i, j) in mines:
                    row_chars.append("X")
                else:
                    val = field[i, j]
                    row_chars.append(str(val) if val > 0 else ".")
            else:
                # 未開示セル：GBMCによる「気配」を表示
                # ここがBuffer(B)の可視化
                res = calculate_gbmc(prev, grid, field, i, j)
                total_q += res.Q
                if res.M > 0: active_m += 1
                row_chars.append(score_to_char(res.D))
        lines.append("".join(row_chars))

    # Header
    print(f"=== KQI: GSMC Integrated LifeSweeper ===")
    print(f"Tick: {tick:<5} | Total Energy(Q): {total_q:.2f} | Motion Active: {active_m}")
    print(f"Params: Theta={THETA}, Mode={K_MODE}")
    if last_msg: print(f"MSG: {last_msg}")
    print("-" * W)
    
    # Map
    print("\n".join(lines))
    print("-" * W)
    print("cmd: open r c | step n | auto | quit")

# -------------------------
# Main Loop
# -------------------------
def main():
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)

    mines, field = make_mines()
    visible = np.zeros((H, W), dtype=bool)

    # LifeGame初期化
    grid = np.random.randint(0, 2, (H, W)).astype(int)
    prev = grid.copy()

    tick = 0
    last_msg = ""
    auto_mode = False

    while True:
        # Render
        render(prev, grid, visible, mines, field, tick, last_msg)
        
        # Input / Wait
        if auto_mode or AUTO_OPEN_EVERY > 0:
            time.sleep(TICK_SEC)
            cmd = "" # オート時は入力スキップ
        else:
            cmd = input("> ").strip().lower()

        # Command Processing
        if cmd in ["q", "quit"]: break
        
        if cmd.startswith("open"):
            try:
                _, r, c = cmd.split()
                boom, stat = open_cell(visible, mines, field, int(r), int(c))
                last_msg = stat
                if boom:
                    render(prev, grid, visible, mines, field, tick, "💥 IMPACT DETECTED 💥")
                    break
            except: last_msg = "err"
            
        elif cmd.startswith("step"):
            # just advance
            pass
            
        elif cmd == "auto":
            auto_mode = not auto_mode
            last_msg = f"Auto: {auto_mode}"

        # LifeGame Update
        prev = grid
        grid = life_step(grid)
        tick += 1

        # Logic: Auto Open (Simulation)
        if auto_mode and tick % 5 == 0:
             # ランダムに開ける（デモ用）
            c = pick_auto_open_candidate(visible, field)
            if c:
                boom, stat = open_cell(visible, mines, field, c[0], c[1])
                last_msg = f"Auto {c} -> {stat}"
                if boom:
                    render(prev, grid, visible, mines, field, tick, "💥 IMPACT DETECTED 💥")
                    break

if __name__ == "__main__":
    main() 