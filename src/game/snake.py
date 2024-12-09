from typing import Tuple, List

from src.game.models.directions import Direction


class Snake:
    def __init__(self, id: int, size: int, start_pos: Tuple[int, int]) -> None:
        self.id: int = id
        self.movement_direction = (
            Direction.RIGHT
            if id % 2 == 0
            else Direction.LEFT
        )
        self.size: int = size
        self.kills: int = 0
        self.red_apples_eaten: int = 0
        self.green_apples_eaten: int = 0
        self.alive: bool = True
        self.head: Tuple[int, int] = start_pos
        self.body: List[Tuple[int, int]] = [self.head]
        self.directions_map: dict[int, Direction] = {
            1: Direction.UP,
            2: Direction.DOWN,
            3: Direction.LEFT,
            4: Direction.RIGHT,
        }
        self.initialize()

    def initialize(self) -> None:
        for body in range(1, self.size):
            snake_segment: Tuple[int, int] = (
                self.head[0],
                self.head[1] - body
                if self.id % 2 == 0
                else self.head[1] + body,
            )
            self.body.append(snake_segment)

    @property
    def direction_one_hot(self) -> List[int]:
        mapping = {
            Direction.UP: [1, 0, 0, 0],
            Direction.DOWN: [0, 1, 0, 0],
            Direction.LEFT: [0, 0, 1, 0],
            Direction.RIGHT: [0, 0, 0, 1],
        }
        return mapping[self.movement_direction]
    
    @property
    def possible_directions(self) -> List[int]:
        """Returns possible directions that the snake can move in."""
        # Return a one-hot encoded list of the possible directions.
        return [
            1 if self.directions_map[i] != self.movement_direction.opposite else 0
            for i in range(1, 5)
        ]

    def snake_controller(self, direction: list[int]) -> None:
        """
        Change the snake's direction based on keyboard input
        or neural network output.

        Args:
            direction list[int]: The direction to move the
                snake in one_hot encoding.
        """
        for i, d in enumerate(direction):
            if d == 1:
                self.movement_direction = self.directions_map[i + 1]

    def move(self) -> List[Tuple[int, int]]:
        """
        Move the snake in the current direction.

        Returns:
            List[Tuple[int, int]]: The area no longer
                occupied by the snake.
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
        if self.size == 0:
            self.alive = False
