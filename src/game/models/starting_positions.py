from typing import Tuple
from dataclasses import dataclass

from src.game.models.directions import Direction


@dataclass
class StartingPositions:
    def __init__(self, map_size: Tuple[int, int]):
        self.height: int = map_size[0]
        self.width: int = map_size[1]

    @property
    def positions(self) -> dict[int, Tuple[int, int]]:
        return {
            # Top left
            0: (3, 3),
            # Top right
            1: (3, self.width - 4),
            # Bottom left
            2: (self.height - 4, 3),
            # Bottom right
            3: (self.height - 4, self.width - 4),
        }

    @property
    def directions(self) -> dict[int, Direction]:
        return {
            0: Direction.RIGHT,
            1: Direction.LEFT,
            2: Direction.RIGHT,
            3: Direction.LEFT,
        }
