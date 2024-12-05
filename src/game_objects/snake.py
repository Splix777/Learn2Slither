from enum import Enum
from typing import Dict, List

from src.utils.config import Config, get_config
from src.game_objects.map import Map


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    def __init__(self, dx: int, dy: int):
        self.dx = dx
        self.dy = dy

    def move(self, x: int, y: int) -> tuple[int, int]:
        """Move in the direction from the current coordinates."""
        return x + self.dx, y + self.dy


class Snake:
    def __init__(self, config: Config, game_map: Map):
        self.config: Config = config
        self.alive: bool = True
        self.map: Map = game_map
        self.head: Dict[str, int] = {}
        self.body: List[Dict[str, int]] = []
        self.size: int = config.snake.start_size
        self.color: str = config.snake.start_color
        self.speed: int = config.snake.start_speed
        self.vision: int = config.snake.start_vision
        self.create_snake()
        self.map.add_apples(self.map.map)

    def create_snake(self) -> None:
        """
        Create the snake with the given configuration.
        """
        # Initialize head position and place it on the map
        self.head = self.map.get_random_empty_position(self.map.map)
        self.map.map[self.head["y"]][self.head["x"]] = (
            self.config.ASCII.snake_head
        )
        self.body.append(self.head)

        for _ in range(self.size - 1):
            self._add_body()

    def _add_body(self) -> None:
        """Add a new body part behind the last segment."""
        last_part = self.body[-1]
        x, y = last_part["x"], last_part["y"]

        for direction in Direction:
            new_x, new_y = direction.move(x, y)
            if self.map.is_empty(new_x, new_y):
                new_body_part = {"x": new_x, "y": new_y}
                self.body.append(new_body_part)
                self.map.map[new_y][new_x] = self.config.ASCII.snake_body
                return
        raise ValueError("No empty spaces for the snake body.")

    def is_body(self, x: int, y: int) -> bool:
        """
        Check if a given position on the map is a snake body part.
        """
        return self.map.map[y][x] == self.config.ASCII.snake_body

    def move(self, new_head: tuple[int, int]) -> None:
        """
        Move the snake to the new position.
        """
        x, y = new_head

        if self.map.is_apple(x, y):
            self._eat_apple(x, y)
        elif self.map.is_wall(x, y):
            self.alive = False
        elif self.is_body(x, y):
            self.alive = False
        else:
            self._move_snake(x, y)

    def _eat_apple(self, x: int, y: int) -> None:
        """
        Eat an apple and grow the snake.
        """
        if self.map.is_green_apple(x, y):
            self.size += 1
            self.map.green_apples -= 1
        elif self.map.is_red_apple(x, y):
            self.size -= 1
            self.map.red_apples -= 1

        self.map.map[y][x] = self.config.ASCII.snake_head
        self.head = {"x": x, "y": y}

        self._add_body()

    def _move_snake(self, x: int, y: int) -> None:
        """
        Move the snake to the new position.
        The body follows the head.
        """
        # Move the body first (from tail to head)
        for i in range(len(self.body) - 1, 0, -1):
            self.body[i]["x"], self.body[i]["y"] = self.body[i - 1]["x"], self.body[i - 1]["y"]
            self.map.map[self.body[i]["y"]][self.body[i]["x"]] = self.config.ASCII.snake_body

        # Move the head to the new position
        self.head = {"x": x, "y": y}
        self.map.map[self.head["y"]][self.head["x"]] = self.config.ASCII.snake_head

        # Clear the old body (where the snake used to be)
        last_part = self.body[-1]
        self.map.map[last_part["y"]][last_part["x"]] = self.config.ASCII.empty

        # Update the body list to reflect the movement
        self.body[0] = self.head
        self.body = self.body[:-1]


if __name__ == "__main__":
    config: Config = get_config()
    game_map: Map = Map(config)
    snake: Snake = Snake(config, game_map)

    game_map.display_map()

    snake.move(Direction.RIGHT.move(*snake.head.values()))

    game_map.display_map()

