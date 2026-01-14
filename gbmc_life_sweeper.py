# gbmc_life_sweeper.py
# LifeGame × Minesweeper → GBMC map (dynamic ASCII output)
# based on: lifegame rules + minesweeper mines/adjacent numbers
# (see uploaded refs) :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

import os
import random
import time
from dataclasses import dataclass

import numpy as np

# -------------------------
# Config
# -------------------------
H, W = 20, 40
MINES = 120            # mine count (tune)
TICK_SEC = 0.15
SEED = None            # set int for reproducibility, e.g. 42
AUTO_OPEN_EVERY = 0    # 0 disables; if >0 auto-open a safe-ish cell every N ticks

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
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if inb(ni, nj):
                yield ni, nj

# -------------------------
# Minesweeper layer
# -------------------------
def make_mines():
    coords = [(i, j) for i in range(H) for j in range(W)]
    mines = set(random.sample(coords, MINES))
    field = np.zeros((H, W), dtype=int)
    field[:] = 0
    for (i, j) in mines:
        field[i, j] = -1
    # adjacent numbers
    for i in range(H):
        for j in range(W):
            if field[i, j] == -1:
                continue
            c = 0
            for ni, nj in neighbors8(i, j):
                if (ni, nj) in mines:
                    c += 1
            field[i, j] = c
    return mines, field

# -------------------------
# Life layer
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
# GBMC scoring (per-cell)
# -------------------------
@dataclass
class GBMC:
    G: float
    B: float
    M: float
    C: float
    S: float  # combined score

def gbmc_cell(prev_grid, grid, i, j) -> GBMC:
    # Local neighbor count
    n = 0
    same = 0
    for ni, nj in neighbors8(i, j):
        n += grid[ni, nj]
        if grid[ni, nj] == grid[i, j]:
            same += 1

    # G: "goal pull" toward life-stability target neighbor count ~2.5
    # closer to 2 or 3 neighbors => higher
    g = 1.0 - min(abs(n - 2.5) / 2.5, 1.0)

    # B: "balance" = will this cell be stable under classic Life rule?
    alive = grid[i, j] == 1
    stable = (alive and (n == 2 or n == 3)) or ((not alive) and n != 3)
    b = 1.0 if stable else 0.0

    # M: "motion" = did it change this tick?
    m = 1.0 if prev_grid[i, j] != grid[i, j] else 0.0

    # C: "coherence" = neighborhood agreement (0..1)
    c = same / 8.0

    # Combined score: tweak weights however you like
    s = 0.35 * g + 0.25 * b + 0.25 * m + 0.15 * c
    return GBMC(G=g, B=b, M=m, C=c, S=s)

def score_to_char(s: float) -> str:
    # low -> high density
    ramp = " .:-=+*#%@"
    idx = int(round(s * (len(ramp) - 1)))
    idx = max(0, min(len(ramp) - 1, idx))
    return ramp[idx]

# -------------------------
# Gameplay helpers
# -------------------------
def open_cell(visible, mines, field, r, c):
    if not inb(r, c):
        return False, "out_of_bounds"
    if visible[r, c]:
        return False, "already_open"
    visible[r, c] = True
    if (r, c) in mines:
        return True, "boom"
    return False, "ok"

def pick_auto_open_candidate(visible, field):
    # pick unopened with low adjacent number (safer vibe) if possible
    candidates = np.argwhere(~visible)
    if len(candidates) == 0:
        return None
    # sort by field number (but mines are hidden; field has -1 where mines are)
    # we avoid obvious mines by skipping -1 (still "cheating" because we know field internally;
    # set AUTO_OPEN_EVERY=0 if you want pure manual)
    safe = [(r, c) for r, c in candidates if field[r, c] >= 0]
    if not safe:
        return None
    safe.sort(key=lambda rc: field[rc[0], rc[1]])
    return safe[0]

# -------------------------
# Render
# -------------------------
def render(prev_grid, grid, visible, mines, field, tick, last_msg=""):
    clear()

    # global stats
    alive_ratio = float(grid.mean())
    opened = int(visible.sum())
    unopened = H * W - opened
    mine_density = MINES / (H * W)
    # simple "equilibrium" target: lower if mines are many
    g_eq = max(0.10, 0.30 - 0.25 * mine_density)
    e = alive_ratio - g_eq

    print(f"GBMC LifeSweeper | tick={tick} | alive={alive_ratio:.3f} | opened={opened}/{H*W} | mines={MINES} | g_eq={g_eq:.3f} | e=alive-g_eq={e:+.3f}")
    if last_msg:
        print(f"msg: {last_msg}")
    print("-" * W)

    # map
    lines = []
    for i in range(H):
        row_chars = []
        for j in range(W):
            if visible[i, j]:
                if (i, j) in mines:
                    row_chars.append("💣")
                else:
                    row_chars.append(str(field[i, j]) if field[i, j] > 0 else " ")
            else:
                gb = gbmc_cell(prev_grid, grid, i, j)
                # If you want GBMC components exposed, you could colorize; we stay ASCII-simple.
                row_chars.append(score_to_char(gb.S))
        lines.append("".join(row_chars))
    print("\n".join(lines))
    print("-" * W)
    print("commands: open r c | step n | help | quit")
    print("tip: r,c are 0-indexed. Example: open 3 12")

# -------------------------
# Main
# -------------------------
def main():
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)

    mines, field = make_mines()
    visible = np.zeros((H, W), dtype=bool)

    grid = np.random.randint(0, 2, (H, W)).astype(int)
    prev = grid.copy()

    tick = 0
    last_msg = ""

    # initial render
    render(prev, grid, visible, mines, field, tick, last_msg)

    while True:
        # one life tick
        prev = grid
        grid = life_step(grid)
        tick += 1

        # optional auto-open
        if AUTO_OPEN_EVERY > 0 and tick % AUTO_OPEN_EVERY == 0:
            cand = pick_auto_open_candidate(visible, field)
            if cand is not None:
                boom, status = open_cell(visible, mines, field, int(cand[0]), int(cand[1]))
                last_msg = f"auto_open {cand[0]} {cand[1]} -> {status}"
                if boom:
                    render(prev, grid, visible, mines, field, tick, last_msg)
                    print("💥 BOOM（責任発生）")
                    return

        render(prev, grid, visible, mines, field, tick, last_msg)
        last_msg = ""

        # command prompt (non-blocking would be nicer, but portability wins)
        cmd = input("> ").strip()
        if not cmd:
            # empty => continue simulation
            continue
        if cmd in ("q", "quit", "exit"):
            return
        if cmd in ("h", "help"):
            last_msg = "open r c / step n / quit"
            continue
        if cmd.startswith("step "):
            try:
                n = int(cmd.split()[1])
                for _ in range(n):
                    prev = grid
                    grid = life_step(grid)
                    tick += 1
                last_msg = f"stepped {n}"
            except Exception:
                last_msg = "step usage: step 10"
            continue
        if cmd.startswith("open "):
            try:
                _, rs, cs = cmd.split()
                r, c = int(rs), int(cs)
                boom, status = open_cell(visible, mines, field, r, c)
                last_msg = f"open {r} {c} -> {status}"
                if boom:
                    render(prev, grid, visible, mines, field, tick, last_msg)
                    print("💥 BOOM（責任発生）")
                    return
            except Exception:
                last_msg = "open usage: open 3 12"
            continue

        last_msg = "unknown command. type: help"

if __name__ == "__main__":
    main()
