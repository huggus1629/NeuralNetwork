import pygame as pg
from grid import SquareGrid
import sys
from os.path import abspath, dirname, join
ROOT = abspath(join(dirname(__file__), '..'))
sys.path.append(ROOT)
from neural_network import NeuralNetwork
from texttowindow import putText, Anchor
import traceback

try:
    # Initialize pg
    pg.init()

    # Screen dimensions
    WIDTH, HEIGHT = 850, 600

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
    grid_origin = (h_padding, (HEIGHT - grid_wh_px) // 2)

    grid = SquareGrid(grid_w, *grid_origin, cell_size)
    previous_cells = []

    nn = NeuralNetwork([784, 64, 32, 10])
    nn.load(f"{ROOT}/Neural_Network.npz") # type: ignore
    update_nn = True
    nn_input = [0] * (grid_w * grid_h)
    nn_output: tuple[float | str, ...] = tuple(1/10 for _ in range(10))

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
                if not grid.mouseInGrid(event.pos):
                    continue
                if event.buttons[0]:  # LMB draws
                    grid.draw(event.pos)
                elif event.buttons[2]:  # RMB erases
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

        grid_empty = all(cell.value == 0 for cell in grid)

        if update_nn and not grid_empty:
            # convert grid to expected format
            nn_input = [float(cell.value) for cell in grid]
            # run input through nn
            nn_output = tuple(float(p[0]) for p in nn.forward(nn_input)) # type: ignore
        update_nn = False

        if grid_empty:
            nn_output = tuple("--" for _ in range(10))

        # Draw everything
        screen.fill(BG)  # Clear the screen

        # loop through grid
        for cell in grid:
            pg.draw.rect(screen, BG.lerp(WHITE, cell.value), cell.rect)
        
        # draw bounding box
        grid_rect = pg.Rect(*grid_origin, grid_wh_px, grid_wh_px)
        pg.draw.rect(screen, WHITE, grid_rect, 2)

        # generate text with all probabilities
        probs_text = ""
        for i, p in enumerate(nn_output):
            percentage_or_str = f"{100 * p:.3f}" if isinstance(p, float) else p
            probs_text += f"{i}:\t{percentage_or_str} %" + ("\n" if i < 9 else "")
        probs_text = probs_text.expandtabs()
        max_: int | str = nn_output.index(max(nn_output)) if not grid_empty else ""
        # render text
        max_index = max_ if isinstance(max_, int) else None
        end_of_text = putText(probs_text,
                              28,
                              Anchor("tl", (grid_rect.right + h_padding, v_padding)),
                              screen,
                              line_spacing=4,
                              hl_line=max_index)

        # Draw a square between probs_text and help_text
        square_x = grid_rect.right + h_padding
        square_size = WIDTH - h_padding - square_x
        square_y = end_of_text.bottom + v_padding
        pg.draw.rect(screen, WHITE, (square_x, square_y, square_size, square_size), 1)
        # print the predicted digit in the center of the square
        putText(str(max_),
                int(1.1*square_size),
                Anchor("c", (square_x + square_size // 2, square_y + square_size // 2)),
                screen)
        
        # render help text
        help_text = """\
Left click + drag do draw
Right click + drag to erase
Double right click to clear\
"""
        putText(help_text,
                22,
                Anchor("bl", (grid_rect.right + h_padding, HEIGHT - v_padding)),
                screen)

        # Update the display
        pg.display.flip()

        # Cap the frame rate
        clock.tick(60)
except Exception:
    traceback.print_exc()
finally:
    # Quit pg
    pg.quit()
    raise SystemExit