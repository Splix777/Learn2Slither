from typing import Tuple, List

from src.utils.directions import Direction


class Snake:
    def __init__(self, id: int, size: int, start_pos: Tuple[int, int]) -> None:
        self.id: int = id
        self.movement_direction = (
            Direction.RIGHT if id % 2 == 0 else Direction.LEFT
        )
        self.size: int = size
        self.kills: int = 0
        self.alive: bool = True
        self.head: Tuple[int, int] = start_pos
        self.body: List[Tuple[int, int]] = [self.head]
        self.directions_map: dict[int, Direction] = {
            0: Direction.UP,
            1: Direction.DOWN,
            2: Direction.LEFT,
            3: Direction.RIGHT,
        }
        self.initialize()

    def initialize(self) -> None:
        for body in range(1, self.size):
            snake_segment: Tuple[int] = (
                self.head[0],
                self.head[1] - body
                if self.id % 2 == 0
                else self.head[1] + body,
            )
            self.body.append(snake_segment)

    def change_direction(self, direction: int) -> None:
        """
        Change the snake's direction based on keyboard input
        or neural network output.

        Args:
            direction (int): The direction to change to.
        """
        if (
            direction in self.directions_map
            and self.directions_map[direction] 
            != self.movement_direction.opposite
        ):
            self.movement_direction: Direction = self.directions_map[direction]

    def move(self) -> List[Tuple[int, int]]:
        """
        Move the snake in the current direction.

        Returns:
            List[Tuple[int, int]]: The area no longer occupied by the snake.
        """
        removed_tails: List[Tuple[int, int]] = []
        if not self.alive:
            return removed_tails

        new_head = (
            self.head[0] + self.movement_direction.value[0],
            self.head[1] + self.movement_direction.value[1],
        )

        self.body.insert(0, new_head)
        self.head: Tuple[int, int] = new_head

        while len(self.body) > self.size:
            removed_tails.append(self.body.pop())

        return removed_tails

    def grow(self) -> None:
        """Grow the snake by one unit."""
        self.size += 1

    def shrink(self) -> None:
        """Shrink the snake by one unit."""
        self.size -= 1
