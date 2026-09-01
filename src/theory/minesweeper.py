import random

H, W, MINES = 10, 10, 15

field = [[0]*W for _ in range(H)]
visible = [[False]*W for _ in range(H)]

# 地雷配置
mines = set(random.sample([(i,j) for i in range(H) for j in range(W)], MINES))
for i, j in mines:
    field[i][j] = -1

# 数字計算
for i in range(H):
    for j in range(W):
        if field[i][j] == -1:
            continue
        field[i][j] = sum(
            (ni,nj) in mines
            for ni in range(i-1,i+2)
            for nj in range(j-1,j+2)
            if 0 <= ni < H and 0 <= nj < W
        )

def draw():
    for i in range(H):
        for j in range(W):
            if not visible[i][j]:
                print("□", end="")
            elif field[i][j] == -1:
                print("💣", end="")
            else:
                print(field[i][j], end="")
        print()

while True:
    draw()
    x, y = map(int, input("open (row col): ").split())
    visible[x][y] = True
    if field[x][y] == -1:
        print("💥 BOOM（責任発生）")
        break
