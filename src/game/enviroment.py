import random
from typing import List, Tuple
from itertools import product

from src.config.settings import Config
from src.game.models.starting_positions import StartingPositions
from src.game.snake import Snake


class Enviroment:
    def __init__(self, config: Config):
        self.config: Config = config
        # <-- Map Dimensions -->
        self.height: int = config.map.board_size.height
        self.width: int = config.map.board_size.width
        # <-- Apples -->
        self.current_red_apples: int = 0
        self.current_green_apples: int = 0
        self.max_red_apples: int = config.map.red_apples
        self.max_green_apples: int = config.map.green_apples
        # <-- Snakes -->
        self.num_snakes: int = config.map.snakes
        self.starting_positions = StartingPositions((self.height, self.width))
        # <-- Initialize Map -->
        self.map: List[List[str]] = self.make_map()
        self.snakes: List[Snake] = self.create_snakes(config.snake.start_size)
        self.add_apples()

    # <-- Map generation methods -->
    def make_map(self) -> List[List[str]]:
        """Create a map with the given configuration."""
        empty_map: List[List[str]] = self._create_blank_map()
        self._add_walls(empty_map)
        return empty_map

    def create_snakes(self, size: int = 3) -> List[Snake]:
        """
        Create snakes for the game.

        Returns:
            List[Snake]: A list of snakes.
        """
        snakes: List[Snake] = []
        for snake_num in range(self.num_snakes):
            snake = Snake(
                id=snake_num,
                size=size,
                start_pos=self.starting_positions.positions[snake_num],
                start_dir=self.starting_positions.directions[snake_num],
            )
            if not self.snake_valid_position(snake):
                raise ValueError("Snake not in valid position")
            snakes.append(snake)
            self.update_snake_position(snake=snake)
        return snakes

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

    def snake_valid_position(self, snake: Snake) -> bool:
        """
        Check if the snake is within the map bounds and all segments
        are in empty spaces.

        Args:
            snake (Snake): The snake to check.

        Returns:
            bool: True if the snake is within the map bounds,
            otherwise False.
        """
        return all(
            1 <= segment[0] < self.height - 1
            and 1 <= segment[1] < self.width - 1
            and self.map[segment[0]][segment[1]] == "empty"
            for segment in snake.body
        )

    def _update_snake_rewards(self, snake: Snake, event: str | None) -> None:
        """
        Update the snake's reward based on the event.

        Args:
            snake (Snake): The snake to update.
            event (str): The event that occurred.
        """
        event_rewards: dict[str, int] = {
            "death": self.config.rules.events.death,
            "green_apple": self.config.rules.events.green_apple,
            "red_apple": self.config.rules.events.red_apple,
        }

        if event in event_rewards:
            snake.update_reward(event, event_rewards[event])

    # <-- Apple methods -->
    def add_apples(self) -> None:
        """
        Add apples to random empty spaces on the map.
        """
        while self.current_green_apples < self.max_green_apples:
            x, y = self.get_random_empty_coordinate()
            self.map[y][x] = "green_apple"
            self.current_green_apples += 1

        while self.current_red_apples < self.max_red_apples:
            x, y = self.get_random_empty_coordinate()
            self.map[y][x] = "red_apple"
            self.current_red_apples += 1

    # <-- Collisions -->
    def check_collision(self, snake: Snake, snakes: List[Snake]) -> str | None:
        if self._collided_with_wall(snake):
            self._delete_snake(snake)
            return "death"
        if self._collided_with_snake(snake, snakes):
            self._delete_snake(snake)
            return "death"
        return None

    def _collided_with_wall(self, snake: Snake) -> bool:
        """
        Checks if the snake collided with a wall.

        Args:
            snake (Snake): The snake to check.

        Returns:
            bool: True if the snake collided with a wall, otherwise False.
        """
        return self.map[snake.head[0]][snake.head[1]] == "wall"

    def _collided_with_snake(self, snake: Snake, snakes: List[Snake]) -> bool:
        """
        Checks if the snake collided with another snake or itself. If
        the snake collided with another snake, the other snake's kills
        count is incremented.

        Args:
            snake (Snake): The snake to check.
            snakes (List[Snake]): List of all snakes in the game.

        Returns:
            bool: True if the snake collided with another snake
                or itself, otherwise False.
        """
        cell: str = self.map[snake.head[0]][snake.head[1]]
        if cell not in ["snake_body", "snake_head"]:
            return False
        for other_snake in snakes:
            if snake.head in other_snake.body and snake.id != other_snake.id:
                other_snake.kills += 1
        return True

    def _delete_snake(self, snake: Snake) -> None:
        """
        Delete a snake from the game.
        
        Args:
            snake (Snake): The snake to delete.
        """
        for segment in range(1, len(snake.body)):
            self.map[snake.body[segment][0]][snake.body[segment][1]] = "empty"
        snake.alive = False

    # <-- Game state update methods -->
    def check_apple_eaten(self, snake: Snake) -> str | None:
        if self.map[snake.head[0]][snake.head[1]] == "green_apple":
            self.current_green_apples -= 1
            snake.green_apples_eaten += 1
            snake.grow()
            return "green_apple"
        elif self.map[snake.head[0]][snake.head[1]] == "red_apple":
            self.current_red_apples -= 1
            snake.red_apples_eaten += 1
            snake.shrink()
            return "red_apple"
        return None

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

    def iterate_snakes(self, actions: List[List[int]] | List[int]) -> None:
        """Perform one step in the game based on the chosen action."""
        [snake.reset_reward() for snake in self.snakes]
        for snake, action in zip(self.snakes, actions):
            snake.snake_controller(action)
            self.move_snake(snake)
            collision_event = self.check_collision(snake, self.snakes)
            apple_event = self.check_apple_eaten(snake)
            if collision_event or apple_event:
                self._update_snake_rewards(
                    snake,
                    collision_event or apple_event)
            self.update_snake_position(snake)

        self.add_apples()

    @property
    def state(self) -> List[List[str]]:
        return self.map
    
    @property
    def snakes_alive(self) -> List[Snake]:
        return [snake for snake in self.snakes if snake.alive]
    
    @property
    def rewards(self) -> List[int]:
        return [snake.reward for snake in self.snakes]