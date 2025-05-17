import pygame as pg
from grid import SquareGrid

# Initialize pg
pg.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600

# Create the screen
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("NN Number Recognition")

# Clock for controlling the frame rate
clock = pg.time.Clock()

# color macros
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BG = (44, 44, 44)

# variables
grid_wh_px = 560
grid_w, grid_h = 28, 28
cell_size = grid_wh_px // grid_w
h_padding = 20
v_padding = 20

grid = SquareGrid(grid_w, h_padding, v_padding, cell_size)
previous_cells = []

update_nn = False

font = pg.font.Font(None, 22)

# Main loop
running = True
while running:
    # Handle events
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

        # detect mouse dragging
        if event.type == pg.MOUSEMOTION:
            if not any(event.buttons):
                continue
            if event.buttons[0]:
                grid.draw(event.pos)
            elif event.buttons[2]:
                grid.draw(event.pos, erase=True)
            update_nn = True

        # detect double right click (clear grid)
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 3:
            if not grid.last_rmb:
                grid.last_rmb = pg.time.get_ticks()
                continue
            now = pg.time.get_ticks()
            if now - grid.last_rmb < 500:
                grid.clear()
                update_nn = True
                continue
            grid.last_rmb = now

    # (Add logic here)
    # convert grid to expected format
    nn_grid = [float(cell.value) for cell in grid]
    if update_nn:
        print("update")
    update_nn = False
    # send nn_grid to neural network
    # process output (int[10])

    # Draw everything
    screen.fill(BG)  # Clear the screen
    # (Add drawing code here)

    # loop through grid
    for cell in grid:
        if cell.value == 0:
            pg.draw.rect(screen, WHITE, cell.rect, -1)
        elif cell.value > 0:  # draw solid if value > 0
            pg.draw.rect(screen, tuple(int(cell.value * i) for i in WHITE), cell.rect)
    
    pg.draw.rect(screen, WHITE, (h_padding, v_padding, grid_wh_px, grid_wh_px), 1)

    # render help text
    help_text = """\
Left click + drag do draw
Right click + drag to erase
Double right click to clear\
"""
    lines = help_text.splitlines()
    line_height = font.get_linesize()

    for i, line in enumerate(lines[::-1]):
        text_surface = font.render(line, 1, WHITE)
        text_rect = text_surface.get_rect()
        text_rect.bottomright = (WIDTH - h_padding, HEIGHT - v_padding - i * line_height)
        screen.blit(text_surface, text_rect)

    # Update the display
    pg.display.flip()

    # Cap the frame rate
    clock.tick(60)

# Quit pg
pg.quit()
raise SystemExit