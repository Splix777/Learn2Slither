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

    def __repr__(self) -> str:
        return f"Enviroment - Height: {self.height} - Width: {self.width}"

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

    def step(self, actions: List[List[int]] | List[int]):
        """Perform one step in the game based on the chosen action."""
        [snake.reset_reward() for snake in self.snakes]
        for snake, action in zip(self.snakes, actions):
            snake.snake_controller(action)
            self.move_snake(snake)
            apple_event = self.check_apple_eaten(snake)
            collision_event = self.check_collision(snake, self.snakes)
            if collision_event or apple_event:
                self._update_snake_rewards(
                    snake,
                    collision_event or apple_event)
            self.update_snake_position(snake)

        self.add_apples()

        return self.rewards, self.snakes_done, self.snake_sizes

    # <-- Snake States -->
    def get_state(self, snake: Snake) -> List[float]:
        """
        Get an enhanced state representation for the snake.

        Args:
            snake (Snake): The snake to compute the state for.
        """
        head_x, head_y = snake.head

        # Snake's current direction as one-hot encoding
        current_direction = snake.direction_one_hot
        # Possible direction based on current direction
        possible_directions = snake.possible_directions_one_hot

        # Distances to nearest apples and walls in each cardinal direction
        green_apples = self.target_distances(head_x, head_y, "green_apple")
        red_apples = self.target_distances(head_x, head_y, "red_apple")
        walls = self.target_distances(head_x, head_y, "wall")
        snake_body = self.target_distances(head_x, head_y, "snake_body")

        # Immediate danger flags
        immediate_danger = self.get_immediate_danger(head_x, head_y)

        return (
            current_direction
            + possible_directions
            + immediate_danger
            + green_apples
            + walls
            + snake_body
            + red_apples
        )

    def target_distances(self, x: int, y: int, target: str) -> List[float]:
        """
        Calculate the distance to the nearest target object in
        each cardinal direction.

        Args:
            x (int): Snake's head x-coordinate.
            y (int): Snake's head y-coordinate.
            target (str): Target object type (e.g., "green_apple").

        Returns:
            List[float]: Distances to the target object in
                [up, down, left, right], scaled to [0, 1].
        """
        directions = [0] * 4

        # Up
        for i in range(1, x + 1):
            if self.map[x - i][y] == target:
                directions[0] = i
                break

        # Down
        for i in range(1, self.height - x):
            if self.map[x + i][y] == target:
                directions[1] = i
                break

        # Left
        for i in range(1, y + 1):
            if self.map[x][y - i] == target:
                directions[2] = i
                break

        # Right
        for i in range(1, self.width - y):
            if self.map[x][y + i] == target:
                directions[3] = i
                break

        # Normalize distances to [0, 1]
        max_distance: float = max(directions)
        if max_distance > 0:
            directions: List[float] = [
                distance / max_distance for distance in directions
            ]

        return directions

    def get_immediate_danger(self, x, y) -> List[float]:
        collision = ["wall", "snake_body", "snake_head"]
        danger: List[float] = [0] * 4

        # Up
        if x < 0 or self.map[x - 1][y] in collision:
            danger[0] = 1
        # Down
        if (
            x + 1 >= self.height
            or self.map[x + 1][y] in collision
        ):
            danger[1] = 1
        # Left
        if y < 0 or self.map[x][y - 1] in collision:
            danger[2] = 1
        # Right
        if y + 1 >= self.width or self.map[x][y + 1] in collision:
            danger[3] = 1

        return danger

    # <-- Properties -->
    @property
    def snakes_done(self) -> List[bool]:
        return [not snake.alive for snake in self.snakes]
    
    @property
    def rewards(self) -> List[int]:
        return [snake.reward for snake in self.snakes]
    
    @property
    def snake_sizes(self) -> List[int]:
        return [snake.size for snake in self.snakes]

    @property
    def snake_states(self) -> List[List[float]]:
        return [self.get_state(snake) for snake in self.snakes]

    @property
    def snake_state_size(self) -> int:
        return len(self.snake_states[0]) if self.snakes else 0


if __name__ == "__main__":
    from src.config.settings import config
    env = Enviroment(config)
    print(env.snakes_done)
    print(env.rewards)
    print(env.snake_sizes)
    print(env.snake_states)