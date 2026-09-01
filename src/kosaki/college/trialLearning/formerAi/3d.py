import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import tan, radians
import numpy as np

# 3D空間のデータを生成
vertices = (
    (-1, -1, -1),
    (-1, -1,  1),
    (-1,  1,  1),
    (-1,  1, -1),
    ( 1, -1, -1),
    ( 1, -1,  1),
    ( 1,  1,  1),
    ( 1,  1, -1)
)

edges = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (4, 8)
)

colors = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
    (0, 0, 0)
)

# 初期の視点位置
camera_distance = 30.0
camera_rotation = [0, 0]

mouse_x, mouse_y = 0, 0

line_start = (0, 0, 0)
line_end = (0, 0, 9)

def draw_cube():
    glBegin(GL_LINES)
    for edge, color in zip(edges, colors):
        glColor3fv(color)
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_lines():
    glBegin(GL_LINES)
    glColor3f(1.0, 0.0, 0.0)  # 赤色の線
    glVertex3fv(line_start)
    glVertex3fv(line_end)
    glEnd()
    
def set_projection_matrix(fovy, aspect, zNear, zFar):
    f = 1.0 / tan(radians(fovy) / 2.0)
    projection_matrix = np.array([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (zFar + zNear) / (zNear - zFar), -1.0],
        [0.0, 0.0, (2 * zFar * zNear) / (zNear - zFar), 0.0]
    ], dtype=np.float32)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glMultMatrixf(projection_matrix)

def set_modelview_matrix():
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -camera_distance)
    glRotatef(camera_rotation[0], 1, 0, 0)
    glRotatef(camera_rotation[1], 0, 1, 0)

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    set_projection_matrix(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEMOTION:
            # マウスの移動量に応じて視点を変更
                dx, dy = event.rel
                camera_rotation[0] += dy  # 上下の回転
                camera_rotation[1] += dx  # 左右の回転
                mouse_x, mouse_y = pygame.mouse.get_pos()

        line_start = (-2 + (mouse_x / display[0] * 4), 0, 0)
        line_end = (2 + (mouse_x / display[0] * 4), 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_cube()
        draw_lines()
        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()
