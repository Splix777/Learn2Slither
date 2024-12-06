import random
from typing import List

from src.utils.config import Config, get_config


class Map:
    def __init__(self, config: Config):
        self.config: Config = config
        self.height: int = config.map.board_size.height
        self.width: int = config.map.board_size.width
        self.green_apples: int = config.map.starting_green_apples
        self.red_apples: int = config.map.starting_red_apples
        self.current_green_apples: int = 0
        self.current_red_apples: int = 0
        self.map: List[List[str]] = self.make_map()

    def make_map(self) -> List[List[str]]:
        """
        Create a map with the given configuration.
        """
        empty_map: List[List[str]] = self._create_blank_map()
        self._add_walls(empty_map)
        return empty_map

    def get_random_empty_position(self, game_map: List[List[str]]):
        """
        Get a random empty position on the map.
        """
        while True:
            x: int = random.randint(1, self.width - 2)
            y: int = random.randint(1, self.height - 2)
            if game_map[y][x] == self.config.ASCII.empty:
                break
        return {"x": x, "y": y}  

    def _create_blank_map(self) -> List[List[str]]:
        """
        Create a blank map with the given configuration.
        """
        game_map: List[List[str]] = []
        for _ in range(self.height):
            game_map.append([self.config.ASCII.empty] * self.width)
        return game_map

    def _add_walls(self, game_map: List[List[str]]):
        """
        Add walls around the map borders.
        """
        for row in range(self.height):
            for col in range(self.width):
                if (
                    row == 0
                    or row == self.height - 1
                    or col == 0
                    or col == self.width - 1
                ):
                    game_map[row][col] = self.config.ASCII.wall

    # <-- Map Validation Methods -->
    def is_empty(self, x: int, y: int) -> bool:
        """
        Check if a given position on the map is empty.
        """
        return self.map[y][x] == self.config.ASCII.empty

    def is_apple(self, x: int, y: int) -> bool:
        """
        Check if a given position on the map is an apple.
        """
        return self.map[y][x] in [
            self.config.ASCII.red_apple,
            self.config.ASCII.green_apple,
        ]
    
    def is_green_apple(self, x: int, y: int) -> bool:
        """
        Check if a given position on the map is a green apple.
        """
        return self.map[y][x] == self.config.ASCII.green_apple
    
    def is_red_apple(self, x: int, y: int) -> bool:
        """
        Check if a given position on the map is a red apple.
        """
        return self.map[y][x] == self.config.ASCII.red_apple

    def is_wall(self, x: int, y: int) -> bool:
        """
        Check if a given position on the map is a wall.
        """
        return self.map[y][x] == self.config.ASCII.wall

    # <-- Apple Methods -->
    def add_apples(self, game_map: List[List[str]]):
        """
        Add apples to random empty spaces on the map.
        """
        while self.current_green_apples < self.green_apples:
            x, y = self.get_random_empty_position(game_map).values()
            game_map[y][x] = self.config.ASCII.green_apple
            self.current_green_apples += 1

        while self.current_red_apples < self.red_apples:
            x, y = self.get_random_empty_position(game_map).values()
            game_map[y][x] = self.config.ASCII.red_apple
            self.current_red_apples += 1

    # <-- Map Display Methods -->
    def display_map(self) -> None:
        """
        Display the map in the console.
        """
        for row in self.map:
            print("".join(row))


if __name__ == "__main__":
    config: Config = get_config()
    map_maker: Map = Map(config)
    game_map: List[List[str]] = map_maker.map

    for row in game_map:
        print("".join(row))
