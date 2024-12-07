from enum import Enum
from typing import Tuple, List

from src.config.settings import Config
from src.ui.game_textures import GameTextures


class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    @property
    def opposite(self) -> "Direction":
        """Returns the opposite direction."""
        return {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }[self]


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
            

    def move(self) -> List[Tuple[int, int]]:
        """
        Move the snake in the current direction.

        Returns:
            List[Tuple[int, int]]: The area no longer occupied by the snake.
        """
        if not self.alive:
            return self.body

        # Ignore input if trying to move in the opposite direction
        if self.movement_direction == self.movement_direction.opposite:
            return []

        # Calculate new head position
        new_head = (
            self.head[0] + self.movement_direction.value[0],
            self.head[1] + self.movement_direction.value[1],
        )

        # Update snake's body
        self.body.insert(0, new_head)
        self.head = new_head

        # Remove the tail if the snake is longer than its size
        return [self.body.pop()] if len(self.body) > self.size else []
