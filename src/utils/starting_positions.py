from typing import Tuple
from dataclasses import dataclass


@dataclass
class StartingPositions:
    def __init__(self, map_size: Tuple[int, int]):
        self.height: int = map_size[0]
        self.width: int = map_size[1]

    @property
    def positions(self) -> dict[int, Tuple[int, int]]:
        return {
            0: (3, 3),
            1: (3, self.width - 4),
            2: (self.height - 4, 3),
            3: (self.height - 4, self.width - 4),
        }

if __name__ == "__main__":
    starting_positions = StartingPositions((10, 10))
    print(starting_positions.positions)