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
        self.body.append(self.head)  # Add the head to the body list

        # Create the body parts based on the snake's size
        for _ in range(self.size - 1):
            self._add_body()

    def _add_body(self) -> None:
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

    def _move(self, new_head: tuple[int, int]) -> None:
        """
        Move the snake to the new position.
        """
        x, y = new_head

        # Check if the new position is an apple
        if self.map.is_apple(x, y):
            self._eat_apple(x, y)
        elif self.map.is_wall(x, y):
            self._die()
        elif self.is_body(x, y):
            self._die()
        else:
            self._move_snake(x, y)
            





if __name__ == "__main__":
    try:
        config: Config = get_config()
        game_map: Map = Map(config)
        snake: Snake = Snake(config, game_map)

        for row in game_map.map:
            print("".join(row))

    except Exception as e:
        print(e)
