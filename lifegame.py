import os
import time

import numpy as np

H, W = 20, 40
grid = np.random.randint(0, 2, (H, W))

def step(grid):
    new = grid.copy()
    for i in range(H):
        for j in range(W):
            neighbors = np.sum(grid[max(0,i-1):i+2, max(0,j-1):j+2]) - grid[i,j]
            if grid[i,j] == 1 and (neighbors < 2 or neighbors > 3):
                new[i,j] = 0
            elif grid[i,j] == 0 and neighbors == 3:
                new[i,j] = 1
    return new

while True:
    os.system("cls" if os.name == "nt" else "clear")
    print("\n".join("".join("■" if c else " " for c in row) for row in grid))
    grid = step(grid)
    time.sleep(0.15)
