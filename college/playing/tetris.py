import random

import pygame

# 初期設定
WIDTH, HEIGHT = 300, 600
BLOCK_SIZE = 30
GRID_WIDTH = WIDTH // BLOCK_SIZE
GRID_HEIGHT = HEIGHT // BLOCK_SIZE
FPS = 60

# 色定義
COLORS = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 150, 0),
    (0, 0, 255),
    (255, 120, 0),
    (255, 255, 0),
    (180, 0, 255),
    (0, 220, 220)
]

# テトリミノの形状
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[2, 2, 2], [0, 2, 0]],  # T
    [[3, 3], [3, 3]],  # O
    [[4, 4, 0], [0, 4, 4]],  # Z
    [[0, 5, 5], [5, 5, 0]],  # S
    [[6, 0, 0], [6, 6, 6]],  # L
    [[0, 0, 7], [7, 7, 7]]   # J
]

class Tetromino:
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = SHAPES.index(shape) + 1
        self.rotation = 0

    def current_shape(self):
        return self.shape[(self.rotation) % len(self.shape)]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.shape)

def create_grid(locked_pos={}):
    grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    for (y, x), color in locked_pos.items():
        grid[y][x] = color
    return grid

def draw_grid(surface, grid):
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            pygame.draw.rect(surface, COLORS[grid[y][x]], (x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)
    
    for x in range(GRID_WIDTH):
        pygame.draw.line(surface, (128,128,128), (x*BLOCK_SIZE, 0), (x*BLOCK_SIZE, HEIGHT))
    for y in range(GRID_HEIGHT):
        pygame.draw.line(surface, (128,128,128), (0, y*BLOCK_SIZE), (WIDTH, y*BLOCK_SIZE))

def valid_space(shape, grid, x, y):
    for ry, row in enumerate(shape):
        for rx, val in enumerate(row):
            if val:
                new_x = x + rx
                new_y = y + ry
                if (new_x < 0 or new_x >= GRID_WIDTH or
                    new_y >= GRID_HEIGHT or
                    (new_y >= 0 and grid[new_y][new_x])):
                    return False
    return True

def clear_rows(grid):
    cleared = 0
    for y in range(GRID_HEIGHT-1, -1, -1):
        if 0 not in grid[y]:
            cleared += 1
            del grid[y]
            grid.insert(0, [0 for _ in range(GRID_WIDTH)])
    
    return cleared ** 2 * 100

def draw_text(surface, text, size, color, x, y):
    font = pygame.font.SysFont('comicsans', size)
    label = font.render(text, 1, color)
    surface.blit(label, (x - label.get_width()/2, y))

def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH + 150, HEIGHT))
    clock = pygame.time.Clock()
    
    locked_positions = {}
    grid = create_grid(locked_positions)
    
    current_piece = Tetromino(GRID_WIDTH//2-2, 0, random.choice(SHAPES))
    next_piece = Tetromino(GRID_WIDTH//2-2, 0, random.choice(SHAPES))
    
    fall_time = 0
    fall_speed = 0.3
    score = 0
    
    run = True
    while run:
        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        clock.tick()
        
        # 自然落下処理
        if fall_time/1000 > fall_speed:
            fall_time = 0
            current_piece.y += 1
            if not valid_space(current_piece.current_shape(), grid, current_piece.x, current_piece.y):
                current_piece.y -= 1
                for y, row in enumerate(current_piece.current_shape()):
                    for x, val in enumerate(row):
                        if val:
                            locked_positions[(current_piece.y + y, current_piece.x + x)] = current_piece.color
                current_piece = next_piece
                next_piece = Tetromino(GRID_WIDTH//2-2, 0, random.choice(SHAPES))
        
        # イベント処理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_piece.x -= 1
                    if not valid_space(current_piece.current_shape(), grid, current_piece.x, current_piece.y):
                        current_piece.x += 1
                elif event.key == pygame.K_RIGHT:
                    current_piece.x += 1
                    if not valid_space(current_piece.current_shape(), grid, current_piece.x, current_piece.y):
                        current_piece.x -= 1
                elif event.key == pygame.K_DOWN:
                    current_piece.y += 1
                    if not valid_space(current_piece.current_shape(), grid, current_piece.x, current_piece.y):
                        current_piece.y -= 1
                elif event.key == pygame.K_UP:
                    current_piece.rotate()
                    if not valid_space(current_piece.current_shape(), grid, current_piece.x, current_piece.y):
                        current_piece.rotate()
                        current_piece.rotate()
                        current_piece.rotate()
        
        # 描画処理
        win.fill((0,0,0))
        draw_grid(win, grid)
        
        # 現在のブロック描画
        for y, row in enumerate(current_piece.current_shape()):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(win, COLORS[val], ((current_piece.x + x)*BLOCK_SIZE,
                                    (current_piece.y + y)*BLOCK_SIZE,
                                    BLOCK_SIZE, BLOCK_SIZE), 0)
        
        # 次のブロック表示
        draw_text(win, "Next", 30, (255,255,255), WIDTH + 75, 50)
        for y, row in enumerate(next_piece.current_shape()):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(win, COLORS[val],
                    (WIDTH + 50 + x*BLOCK_SIZE,
                    100 + y*BLOCK_SIZE,
                    BLOCK_SIZE, BLOCK_SIZE), 0)
        
        # スコア表示
        draw_text(win, f"Score: {score}", 30, (255,255,255), WIDTH + 75, 300)
        
        pygame.display.update()
        
        # 行消去処理
        cleared = clear_rows(grid, locked_positions)
        score += cleared
        
        # ゲームオーバー判定
        if any(grid[0][x] != 0 for x in range(GRID_WIDTH)):
            draw_text(win, "GAME OVER!", 50, (255,0,0), WIDTH/2, HEIGHT/2)
            pygame.display.update()
            pygame.time.delay(2000)
            run = False

    pygame.quit()

if __name__ == "__main__":
    main()
