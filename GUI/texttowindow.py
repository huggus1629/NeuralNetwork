from pygame.font import Font
from pygame import Color, Surface

anchors = {"tl", "tr", "bl", "br"}

class Anchor:
    def __init__(self, type: str, pos: tuple[int, int]):
        if type not in anchors:
            raise ValueError(f"Anchor.type must be one of {anchors}.")
        self.type = type
        self.pos = pos

def putText(text: str,
            font: Font,
            anchor: Anchor,
            screen: Surface,
            hl_line: int | None = None,
            color: Color = Color(255, 255, 255),
            hl_color: Color = Color(0, 255, 0)):
    
    # dir = True if going from top to bottom line, False otherwise
    dir = True if anchor.type[0] == "t" else False

    text = text.strip()  # clean text
    lines = text.splitlines()
    if hl_line is not None and not (0 <= hl_line < len(lines)):
        raise IndexError(f"Requested line to be highlighted is out of range: {hl_line}.")
    line_height = font.get_linesize()

    x, y = anchor.pos

    for i, line in enumerate(lines[::(1 if dir else -1)]):
        text_surface = font.render(line, 1, Color(0, 0, 0) if i == hl_line else color, hl_color if i == hl_line else None)
        text_rect = text_surface.get_rect()
        new_pos = (x, y + line_height * (i if dir else -i))
        if anchor.type == "tl":
            text_rect.topleft = new_pos
        elif anchor.type == "tr":
            text_rect.topright = new_pos
        elif anchor.type == "bl":
            text_rect.bottomleft = new_pos
        elif anchor.type == "br":
            text_rect.bottomright = new_pos

        screen.blit(text_surface, text_rect)