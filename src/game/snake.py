from typing import Tuple, List

from src.utils.directions import Direction
from src.config.settings import Config
from src.ui.game_textures import GameTextures


class Snake:
    def __init__(
        self,
        config: Config,
        textures: GameTextures,
        start_position: Tuple[int, int],
        id: int,
    ):
        self.config = config
        self.textures = textures
        self.id = id
        # Snake index 0 and 2 start moving right, 1 and 3 start moving left
        self.movement_direction = (
            Direction.RIGHT if id % 2 == 0 else Direction.LEFT
        )
        self.kills = 0
        self.alive = True
        self.head = start_position
        self.body = [self.head]
        self.size = config.snake.start_size
        self.speed = config.snake.start_speed
        self.directions_map = {
            "W": Direction.UP,
            "S": Direction.DOWN,
            "A": Direction.LEFT,
            "D": Direction.RIGHT,
        }
        self.initialize()

    def initialize(self) -> None:
        for body in range(1, self.size):
            new_part = (
                self.head[0],
                self.head[1] - body
                if self.id % 2 == 0
                else self.head[1] + body,
            )
            self.body.append(new_part)

    def change_direction(self, direction: str | int) -> None:
        """
        Change the snake's direction based on keyboard input or neural network output.

        Args:
            direction (str or int): Direction to change to (can be a string or integer).
        """
        if isinstance(direction, str):
            # If the direction is a string ('W', 'S', 'A', 'D')
            if (
                direction in self.directions_map
                and self.directions_map[direction]
                != self.movement_direction.opposite
            ):
                self.movement_direction = self.directions_map[direction]

        elif isinstance(direction, int):
            # If the direction is an integer (0, 1, 2, 3)
            # Map the integer to the corresponding keyboard direction
            direction_map = {0: "W", 1: "S", 2: "A", 3: "D"}
            keyboard_direction = direction_map.get(direction)
            if (
                keyboard_direction
                and self.directions_map[keyboard_direction]
                != self.movement_direction.opposite
            ):
                self.movement_direction = self.directions_map[
                    keyboard_direction
                ]

    def move(self) -> List[Tuple[int, int]]:
        """
        Move the snake in the current direction.

        Returns:
            List[Tuple[int, int]]: The area no longer occupied by the snake.
        """
        removed_tails: List[Tuple[int, int]] = []
        if not self.alive:
            return removed_tails

        # Calculate new head position
        new_head = (
            self.head[0] + self.movement_direction.value[0],
            self.head[1] + self.movement_direction.value[1],
        )

        # Update snake's body
        self.body.insert(0, new_head)
        self.head: Tuple[int, int] = new_head

        # Add or remove tails to match the snake's size
        while len(self.body) > self.size:
            removed_tails.append(self.body.pop())

        return removed_tails

    def grow(self) -> None:
        """Grow the snake by one unit."""
        self.size += 1

    def shrink(self) -> None:
        """Shrink the snake by one unit."""
        self.size -= 1
