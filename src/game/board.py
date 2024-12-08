import random
from typing import List, Tuple
from itertools import product

from src.config.settings import Config, get_config
from src.ui.game_textures import GameTextures
from src.game.snake import Snake


class GameBoard:
    def __init__(self, config: Config, textures: GameTextures):
        self.config: Config = config
        # self.textures: GameTextures = textures
        # self.console = Console()
        self.height: int = config.map.board_size.height
        self.width: int = config.map.board_size.width
        # 4 cardinal directions not including head
        self.snake_vision: int = (2 * self.height + 2 * self.width - 4) // 2
        self.green_apples: int = config.map.green_apples
        self.red_apples: int = config.map.red_apples
        self.current_green_apples: int = 0
        self.current_red_apples: int = 0
        # <-- Snakes -->
        self.num_snakes: int = config.map.snakes
        # self.snakes: List[Snake] = []
        self.snake_starting_positions = {
            0: (6, 6),
            1: (1, self.width - 4),
            2: (self.height - 2, 3),
            3: (self.height - 2, self.width - 4),
        }
        # <-- Initialize Map -->
        self.map: List[List[str]] = self.make_map()

    # <-- Map generation methods -->
    def make_map(self) -> List[List[str]]:
        """
        Create a map with the given configuration.
        """
        empty_map: List[List[str]] = self._create_blank_map()
        self._add_walls(empty_map)
        return empty_map

    def _create_blank_map(self) -> List[List[str]]:
        """
        Create a blank map with the given configuration.

        Returns:
            List[List[str]]: A blank game map.
        """
        return [["empty"] * self.width for _ in range(self.height)]

    def _add_walls(self, game_map: List[List[str]]) -> None:
        """
        Add walls around the map borders.

        Args:
            game_map (List[List[str]]): The game map.
        """
        for row, col in product(range(self.height), range(self.width)):
            if (
                row == 0
                or row == self.height - 1
                or col == 0
                or col == self.width - 1
            ):
                game_map[row][col] = "wall"

    # <-- Utility methods -->
    def get_random_empty_coordinate(self) -> Tuple[int, int]:
        """
        Get a random empty position on the map.

        Returns:
            Tuple[int, int]: The x, y coordinates of the empty position.
        """
        while True:
            x: int = random.randint(1, self.width - 2)
            y: int = random.randint(1, self.height - 2)
            if self.map[y][x] == "empty":
                break
        return x, y

    # <-- Apple methods -->
    def add_apples(self) -> None:
        """
        Add apples to random empty spaces on the map.
        """
        while self.current_green_apples < self.green_apples:
            x, y = self.get_random_empty_coordinate()
            self.map[y][x] = "green_apple"
            self.current_green_apples += 1

        while self.current_red_apples < self.red_apples:
            x, y = self.get_random_empty_coordinate()
            self.map[y][x] = "red_apple"
            self.current_red_apples += 1

    # <-- Create Snakes -->
    def check_collision(self, snake: Snake, all_snakes: List[Snake]) -> bool:
        if self._collided_with_wall(snake):
            self._delete_snake(snake)
            return True
        if self._collided_with_snake(snake, all_snakes):
            self._delete_snake(snake)
            return True
        return False

    def _collided_with_wall(self, snake: Snake) -> bool:
        """
        Checks if the snake collided with a wall.

        Args:
            snake (Snake): The snake to check.

        Returns:
            bool: True if the snake collided with a wall, otherwise False.
        """
        return self.map[snake.head[0]][snake.head[1]] == "wall"

    def _collided_with_snake(self, snake: Snake, all_snakes: List[Snake]) -> bool:
        """
        Checks if the snake collided with another snake or itself.

        Args:
            snake (Snake): The snake to check.
            all_snakes (List[Snake]): List of all snakes in the game.

        Returns:
            bool: True if the snake collided with another snake or itself, otherwise False.
        """
        cell = self.map[snake.head[0]][snake.head[1]]
        if cell not in ["snake_body", "snake_head"]:
            return False

        for other_snake in all_snakes:
            if snake.head in other_snake.body and snake.id != other_snake.id:
                other_snake.kills += 1
                return True

        return True

    def _delete_snake(self, snake: Snake) -> None:
        for segment in range(1, len(snake.body)):
            self.map[snake.body[segment][0]][snake.body[segment][1]] = "empty"

    def check_apple_eaten(self, snake: Snake) -> None:
        if self.map[snake.head[0]][snake.head[1]] == "green_apple":
            self.current_green_apples -= 1
            snake.grow()
        elif self.map[snake.head[0]][snake.head[1]] == "red_apple":
            self.current_red_apples -= 1
            snake.shrink()

    def update_snake_position(self, snake: Snake) -> None:
        if not snake.alive:
            return
        for segment in snake.body:
            if segment == snake.head:
                self.map[segment[0]][segment[1]] = "snake_head"
            else:
                self.map[segment[0]][segment[1]] = "snake_body"

    def move_snake(self, snake: Snake) -> None:
        empty_spaces: List[Tuple[int, int]] = snake.move()
        for space in empty_spaces:
            self.map[space[0]][space[1]] = "empty"


if __name__ == "__main__":
    config: Config = get_config()
    textures: GameTextures = GameTextures(config)
    board: GameBoard = GameBoard(config, textures)

    board.add_apples()
