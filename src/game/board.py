import random
from typing import List, Tuple
from itertools import product

from src.config.settings import Config, get_config
from src.ui.game_textures import GameTextures
from src.game.snake import Snake


class GameBoard:
    def __init__(self, config: Config, textures: GameTextures):
        self.config: Config = config
        self.textures: GameTextures = textures
        self.height: int = config.map.board_size.height
        self.width: int = config.map.board_size.width
        self.green_apples: int = config.map.green_apples
        self.red_apples: int = config.map.red_apples
        self.current_green_apples: int = 0
        self.current_red_apples: int = 0
        # <-- Snakes -->
        self.num_snakes: int = config.map.snakes
        self.snakes: List[Snake] = []
        self.snake_starting_positions = {
            0: (1, 3),
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
    def check_collision(self, snake: Snake) -> bool:
        return self.map[snake.head[1]][snake.head[0]] in [
            "wall",
            "snake_body",
            "snake_head",
        ]

    def update_snake_position(self, snake: Snake) -> None:
        for segment in snake.body:
            if segment == snake.head:
                self.map[segment[0]][segment[1]] = "snake_head"
            else:
                self.map[segment[0]][segment[1]] = "snake_body"


    def render(self) -> None:
        return (
            self._render_ascii()
            if self.textures.gui_mode == "cli"
            else self._render_pygame()
        )

    def _render_ascii(self) -> None:
        from rich.console import Console

        console = Console()

        for row in self.map:
            for cell in row:
                console.print(self.textures.textures[cell], end="")
            console.print()

    def _render_pygame(self) -> None:
        pass


if __name__ == "__main__":
    config: Config = get_config()
    textures: GameTextures = GameTextures(config)
    board: GameBoard = GameBoard(config, textures)

    board.add_apples()
