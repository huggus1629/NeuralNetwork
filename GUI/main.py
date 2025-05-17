import pygame as pg
from grid import SquareGrid
import sys
from os.path import abspath, dirname, join
ROOT = abspath(join(dirname(__file__), '..'))
sys.path.append(ROOT)
from neural_network import NeuralNetwork
from texttowindow import putText, Anchor


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
WHITE = pg.Color(255, 255, 255)
BLACK = pg.Color(0, 0, 0)
BG = pg.Color(44, 44, 44)

# variables
grid_wh_px = 560
grid_w, grid_h = 28, 28
cell_size = grid_wh_px // grid_w
h_padding = 20
v_padding = 20

grid = SquareGrid(grid_w, h_padding, v_padding, cell_size)
previous_cells = []

nn = NeuralNetwork([784, 64, 32, 10])
nn.load(f"{ROOT}/Neural_Network.npz") # type: ignore
update_nn = True
nn_input = [0] * (grid_w * grid_h)
nn_output: tuple[float, ...] = tuple(1/10 for _ in range(10))

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

    if update_nn:
        # convert grid to expected format
        nn_input = [float(cell.value) for cell in grid]
        nn_output = tuple(float(p[0]) for p in nn.forward(nn_input)) # type: ignore
    update_nn = False

    # Draw everything
    screen.fill(BG)  # Clear the screen

    # loop through grid
    for cell in grid:
        pg.draw.rect(screen, BG.lerp(WHITE, cell.value), cell.rect)
    
    # draw bounding box
    pg.draw.rect(screen, WHITE, (h_padding, v_padding, grid_wh_px, grid_wh_px), 1)

    # generate text with all probabilities
    probs_text = ""
    for i, p in enumerate(nn_output):
        probs_text += f"{i}:\t{(100 * p):.3f}%" + ("\n" if i < 9 else "")
    probs_text = probs_text.expandtabs()
    max_index = nn_output.index(max(nn_output))
    # render text
    putText(probs_text,
            font,
            Anchor("tl", (600, v_padding)),
            screen,
            hl_line=max_index)

    # render help text
    help_text = """\
Left click + drag do draw
Right click + drag to erase
Double right click to clear\
"""
    putText(help_text,
            font,
            Anchor("br", (WIDTH - h_padding, HEIGHT - v_padding)),
            screen)

    # Update the display
    pg.display.flip()

    # Cap the frame rate
    clock.tick(60)

# Quit pg
pg.quit()
raise SystemExit