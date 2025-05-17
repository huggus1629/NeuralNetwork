from pygame import Rect


class RectValuePair:
    def __init__(self, rect: Rect, value: float):
        self.rect = rect
        self.value = value
    
    def clear(self):
        self.value = 0

class SquareGrid:
    def __init__(self, s: int, x_offset: int, y_offset: int, cell_size: int, value: int = 0):
        self.sidelen = int(s)

        if self.sidelen <= 0:
            raise ValueError("Grid dimensions must be positive integers.")

        self.x_offset = x_offset
        self.y_offset = y_offset
        self.cell_size = cell_size

        self.cells: list[RectValuePair] = []
        for y in range(self.sidelen):
            for x in range(self.sidelen):
                rect = Rect(x * self.cell_size + self.x_offset,
                            y * self.cell_size + self.y_offset,
                            cell_size + 1,
                            cell_size + 1)
                self.cells.append(RectValuePair(rect, value))
        self.previous_cells: list[RectValuePair | None] = []
        self.previous_cells_erased: list[RectValuePair | None] = []

        self.last_rmb: bool | int = False

    def __getitem__(self, index: int) -> RectValuePair:
        return self.cells[index]
    
    def __setitem__(self, index: int, value: int):
        self.cells[index].value = value

    def __len__(self) -> int:
        return len(self.cells)
    
    def __iter__(self):
        return iter(self.cells)
    
    def clear(self):
        [cell.clear() for cell in self.cells]
        self.previous_cells.clear()
        self.previous_cells_erased.clear()
        self.last_rmb = False

    def mouseInGrid(self, mouse_pos: tuple[int, int]) -> bool:
        x, y = mouse_pos
        return (self.x_offset <= x < self.x_offset + self.sidelen * self.cell_size and
                self.y_offset <= y < self.y_offset + self.sidelen * self.cell_size)

    def indexFrom2D(self, x: int, y: int) -> int:
        if not (0 <= x < self.sidelen and 0 <= y < self.sidelen):
            raise IndexError
        return x + y * self.sidelen
    
    def indexTo2D(self, index: int) -> tuple[int, int]:
        if not 0 <= index < len(self.cells):
            raise IndexError
        y, x = divmod(index, self.sidelen)
        return x, y
    
    def getCellFrom2DIndex(self, x: int, y: int) -> RectValuePair:
        return self.cells[self.indexFrom2D(x, y)]

    def getCellFromMousePos(self, mouse_pos: tuple[int, int]) -> tuple[RectValuePair, int] | tuple[None, None]:
        if not self.mouseInGrid(mouse_pos):
            return None, None

        x, y = mouse_pos
        x -= self.x_offset
        y -= self.y_offset

        grid_x = x // self.cell_size
        grid_y = y // self.cell_size
        
        index = self.indexFrom2D(grid_x, grid_y)
        return self.cells[index], index
    
    def getSurroundingCells(self, index: int, depth: int = 1) -> list[RectValuePair | None]:
        x, y = self.indexTo2D(index)
        
        surrounding: list[RectValuePair | None] = []

        for d in range(1, depth + 1):
            corners = ((x - d, y - d),
                       (x + d, y - d),
                       (x + d, y + d),
                       (x - d, y + d))
            for i in range(4):                
                for j in range(2*d):
                    try:
                        if i == 0:
                            surrounding.append(self.getCellFrom2DIndex(corners[i][0] + j, corners[i][1]))
                        elif i == 1:
                            surrounding.append(self.getCellFrom2DIndex(corners[i][0], corners[i][1] + j))
                        elif i == 2:
                            surrounding.append(self.getCellFrom2DIndex(corners[i][0] - j, corners[i][1]))
                        elif i == 3:
                            surrounding.append(self.getCellFrom2DIndex(corners[i][0], corners[i][1] - j))
                    except IndexError:
                        surrounding.append(None)
        
        return surrounding
    
    def draw(self, mouse_pos: tuple[int, int], erase: bool = False):
        c, index = self.getCellFromMousePos(mouse_pos)
        if c is None or index is None:
            return
        c.value = 1 if not erase else 0
        if c in self.previous_cells and not erase:
            return
        if c in self.previous_cells_erased and erase:
            return

        surrounding = self.getSurroundingCells(index)  # only depth 1 works!
        
        if not erase:
            self.previous_cells = surrounding
            self.previous_cells.append(c)
        else:
            self.previous_cells_erased = surrounding
            self.previous_cells_erased.append(c)

        for i, sc in enumerate(surrounding):
            if sc is None:
                continue
            change = 0.5 if i % 2 else 0.25
            sc.value += change if not erase else -change
            sc.value = max(0, min(sc.value, 1))  # clamp to 0-1