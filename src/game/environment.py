import random
from typing import List, Tuple
from itertools import product
import torch

from src.game.snake import Snake
from src.config.settings import Config
from src.game.models.directions import Direction
from src.game.models.starting_positions import StartingPositions


class Environment:
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
        """Create snakes for the game."""
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
        """Create a blank map with the given configuration."""
        return [["empty"] * self.width for _ in range(self.height)]

    def _add_walls(self, game_map: List[List[str]]) -> None:
        """Add walls around the map borders."""
        for row, col in product(range(self.height), range(self.width)):
            if (
                row == 0
                or row == self.height - 1
                or col == 0
                or col == self.width - 1
            ):
                game_map[row][col] = "wall"

    def reset(self) -> None:
        """Reset the game environment."""
        self.map = self.make_map()
        self.snakes = self.create_snakes(self.config.snake.start_size)
        self.current_red_apples = 0
        self.current_green_apples = 0
        self.add_apples()

    # <-- Utility methods -->
    def get_random_empty_coordinate(self) -> Tuple[int, int]:
        """Get a random empty position on the map."""
        while True:
            x: int = random.randint(1, self.width - 2)
            y: int = random.randint(1, self.height - 2)
            if self.map[y][x] == "empty":
                break
        return x, y

    def snake_valid_position(self, snake: Snake) -> bool:
        """Check if the snake is within the map bounds."""
        return all(
            1 <= segment[0] < self.height - 1
            and 1 <= segment[1] < self.width - 1
            and self.map[segment[0]][segment[1]] == "empty"
            for segment in snake.body
        )

    # <-- Apple methods -->
    def add_apples(self) -> None:
        """Add apples to random empty spaces on the map."""
        while self.current_green_apples < self.max_green_apples:
            x, y = self.get_random_empty_coordinate()
            self.map[y][x] = "green_apple"
            self.current_green_apples += 1

        while self.current_red_apples < self.max_red_apples:
            x, y = self.get_random_empty_coordinate()
            self.map[y][x] = "red_apple"
            self.current_red_apples += 1

    # <-- Events -->
    def check_collision(
        self, snake: Snake, snakes: List[Snake]
    ) -> str | None:
        """Check if the snake collided with a wall or another snake."""
        if self._collided_with_wall(snake):
            self._delete_snake(snake)
            return "death"
        if self._collided_with_snake(snake, snakes):
            self._delete_snake(snake)
            return "death"
        return None

    def _collided_with_wall(self, snake: Snake) -> bool:
        """Checks if the snake collided with a wall."""
        return self.map[snake.head[0]][snake.head[1]] == "wall"

    def _collided_with_snake(self, snake: Snake, snakes: List[Snake]) -> bool:
        """Checks if the snake collided with another snake or itself."""
        cell: str = self.map[snake.head[0]][snake.head[1]]
        if cell not in ["snake_body", "snake_head"]:
            return False
        for other_snake in snakes:
            if snake.head in other_snake.body and snake.id != other_snake.id:
                other_snake.kills += 1
        return True

    def _delete_snake(self, snake: Snake) -> None:
        """Delete a snake from the game."""
        for segment in range(1, len(snake.body)):
            self.map[snake.body[segment][0]][snake.body[segment][1]] = "empty"
        snake.alive = False

    def check_apple_eaten(self, snake: Snake) -> str | None:
        """Check if the snake ate an apple."""
        if self.map[snake.head[0]][snake.head[1]] == "green_apple":
            self.current_green_apples -= 1
            snake.green_apples_eaten += 1
            snake.steps_without_food = 0
            snake.grow()
            return "green_apple"
        elif self.map[snake.head[0]][snake.head[1]] == "red_apple":
            self.current_red_apples -= 1
            snake.red_apples_eaten += 1
            snake.steps_without_food = 0
            snake.shrink()
            return "red_apple"
        return None

    def check_looping(self, snake: Snake) -> str | None:
        """Check if the snake is looping."""
        if snake.steps_without_food >= self.config.rules.steps_no_apple:
            return "looping"
        return None

    def _update_snake_rewards(self, snake: Snake, event: str | None) -> None:
        """Update the snake's reward based on the event."""
        event_rewards: dict[str, int] = {
            "death": self.config.rules.events.death,
            "green_apple": self.config.rules.events.green_apple,
            "red_apple": self.config.rules.events.red_apple,
            "looping": self.config.rules.events.looping,
        }

        if event in event_rewards:
            snake.update_reward(event, event_rewards[event])

    # <-- Snake Movement -->
    def update_snake_position(self, snake: Snake) -> None:
        """Update the snake's position on the map."""
        if not snake.alive:
            return

        for segment in snake.body:
            if segment == snake.head:
                self.map[segment[0]][segment[1]] = "snake_head"
            else:
                self.map[segment[0]][segment[1]] = "snake_body"

    def move_snake(self, snake: Snake) -> None:
        """Move the snake in the current direction."""
        empty_spaces: List[Tuple[int, int]] = snake.move()
        snake.steps_without_food += 1
        for space in empty_spaces:
            self.map[space[0]][space[1]] = "empty"

    def step(self, actions: List[List[int]] | List[int]) -> Tuple:
        """Perform one step in the game based on the chosen action."""
        [snake.reset_reward() for snake in self.snakes]
        for snake, action in zip(self.snakes, actions):
            snake.snake_controller(action)
            self.move_snake(snake)
            apple_event = self.check_apple_eaten(snake)
            collision_event = self.check_collision(snake, self.snakes)
            looping_event = self.check_looping(snake)
            if collision_event or apple_event or looping_event:
                self._update_snake_rewards(
                    snake,
                    collision_event or apple_event or looping_event
                )
            self.update_snake_position(snake)

        self.add_apples()

        return self.rewards, self.snakes_dead, self.snake_sizes

    def process_human_turn(self) -> None:
        """Perform one step in the game based on the chosen action."""
        [snake.reset_reward() for snake in self.snakes]
        for snake in self.snakes:
            self.move_snake(snake)
            apple_event = self.check_apple_eaten(snake)
            collision_event = self.check_collision(snake, self.snakes)
            if collision_event or apple_event:
                self._update_snake_rewards(
                    snake, collision_event or apple_event
                )
            self.update_snake_position(snake)

        self.add_apples()

    # <-- Snake States Getters -->
    def get_state(self, snake: "Snake") -> torch.Tensor:
        """Get an enhanced state representation for the snake."""
        head_x, head_y = snake.head

        # Snake's current direction and possible directions
        current_direction = snake.direction_one_hot
        possible_directions = snake.possible_directions_one_hot

        # Enhanced features
        green_apples = self.target_distances(head_x, head_y, "green_apple")
        red_apples = self.target_distances(head_x, head_y, "red_apple")
        walls = self.target_distances(head_x, head_y, "wall")
        snake_body = self.target_distances(head_x, head_y, "snake_body")
        immediate_danger = self.assess_nearby_risks(
            head_x, head_y
        )
        segmented_vision = self.segmented_vision(head_x, head_y, max_steps=5)
        path_clearances = self.path_clearances(head_x, head_y)

        # Contextual features
        apple_density = self.apple_density(head_x, head_y)
        open_space_ratio = self.open_space_ratio(head_x, head_y)

        # Concatenate all features tensors
        return torch.cat(
            [
                torch.tensor(current_direction),
                torch.tensor(possible_directions),
                green_apples,
                red_apples,
                walls,
                snake_body,
                immediate_danger,
                segmented_vision,
                path_clearances,
                apple_density,
                open_space_ratio,
            ]
        )

    def get_state_by_id(self, snake_id: int) -> torch.Tensor:
        """Get the state of a specific snake."""
        return self.get_state(self.snakes[snake_id])

    # <-- Enhanced Features -->
    def target_distances(self, x: int, y: int, target: str) -> torch.Tensor:
        """Calculate normalized distances to the closest target."""
        distances = torch.ones(4, dtype=torch.float)
        # directions = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])

        # for index, (dx, dy) in enumerate(directions):
        #     steps = 0
        #     while True:
        #         nx, ny = x + steps * dx.item(), y + steps * dy.item()
        #         if (
        #             nx < 0
        #             or ny < 0
        #             or nx >= self.height
        #             or ny >= self.width
        #             or self.map[nx][ny] == "wall"
        #         ):
        #             break
        #         if self.map[nx][ny] == target:
        #             distances[index] = steps / max(self.width, self.height)
        #             break
        #         steps += 1

        # return distances

        for index, direction in enumerate(Direction):
            dr, dc = direction.value
            for step in range(1, max(self.width, self.height)):
                nx, ny = x + dr * step, y + dc * step
                if not (0 <= nx < self.height and 0 <= ny < self.width):
                    distances[index] = self._normalize(
                        step - 1,
                        max(self.width, self.height)
                    )
                    break
                if self.map[nx][ny] == target:
                    distances[index] = self._normalize(
                        step,
                        max(self.width, self.height)
                    )
                    break

        return distances

    def assess_nearby_risks(self, x: int, y: int) -> torch.Tensor:
        """Detect immediate danger in the four cardinal directions."""
        danger = torch.zeros(4, dtype=torch.float)
        # directions = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])

        # for index, (dx, dy) in enumerate(directions):
        #     nx, ny = x + dx.item(), y + dy.item()
        #     if (
        #         nx < 0
        #         or ny < 0
        #         or nx >= self.height
        #         or ny >= self.width
        #         or self.map[nx][ny] in ["wall", "snake_body"]
        #     ):
        #         danger[index] = 1.0

        # return danger

        for index, direction in enumerate(Direction):
            dr, dc = direction.value
            nx, ny = x + dr, y + dc
            if (
                nx < 0
                or ny < 0
                or nx >= self.height
                or ny >= self.width
                or self.map[nx][ny] in ["wall", "snake_body", "snake_head"]
            ):
                danger[index] = 1.0

        return danger

    def segmented_vision(self, x: int, y: int, max_steps: int = 5) -> torch.Tensor:
        """Provide segmented vision looking steps ahead in each direction."""
        vision = torch.ones(4, dtype=torch.float)
        directions = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])

        for index, (dx, dy) in enumerate(directions):
            for step in range(1, max_steps + 1):
                nx, ny = x + step * dx.item(), y + step * dy.item()
                if nx < 0 or ny < 0 or nx >= self.height or ny >= self.width:
                    vision[index] = 1.0
                    break
                cell = self.map[nx][ny]
                if cell == "wall":
                    vision[index] = 1.0
                    break
                elif cell == "snake_body":
                    vision[index] = 0.5
                    break
                elif cell in ["green_apple", "red_apple"]:
                    vision[index] = 0.0
                    break
            else:
                vision[index] = 0.0

        return vision

    def path_clearances(self, x: int, y: int) -> torch.Tensor:
        """Calculate the maximum clear path length in each direction."""
        clearances = torch.zeros(4, dtype=torch.float)
        directions = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])

        for index, (dx, dy) in enumerate(directions):
            steps = 0
            while True:
                nx, ny = (
                    x + (steps + 1) * dx.item(),
                    y + (steps + 1) * dy.item(),
                )
                if (
                    nx < 0
                    or ny < 0
                    or nx >= self.height
                    or ny >= self.width
                    or self.map[nx][ny] in ["wall", "snake_body"]
                ):
                    break
                steps += 1
            clearances[index] = steps / max(self.width, self.height)

        return clearances

    def apple_density(self, x: int, y: int) -> torch.Tensor:
        """Calculate the density of apples in each direction."""
        densities = torch.zeros(4, dtype=torch.float)
        directions = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])

        for index, (dx, dy) in enumerate(directions):
            count = 0
            for step in range(1, 6):
                nx, ny = x + step * dx.item(), y + step * dy.item()
                if nx < 0 or ny < 0 or nx >= self.height or ny >= self.width:
                    break
                if self.map[nx][ny] in ["green_apple", "red_apple"]:
                    count += 1
            densities[index] = count / 5.0

        return densities

    def open_space_ratio(self, x: int, y: int) -> torch.Tensor:
        """Calculate the ratio of open space in each direction."""
        open_ratios = torch.zeros(4, dtype=torch.float)
        directions = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])

        for index, (dx, dy) in enumerate(directions):
            open_count = 0
            for step in range(1, 6):
                nx, ny = x + step * dx.item(), y + step * dy.item()
                if nx < 0 or ny < 0 or nx >= self.height or ny >= self.width:
                    break
                if self.map[nx][ny] == "empty":
                    open_count += 1
            open_ratios[index] = open_count / 5.0

        return open_ratios

    # <-- State Utilities -->
    def _normalize(self, value: int, max_value: int) -> float:
        """Normalize a value between 0 and 1."""
        return value / max_value if max_value != 0 else 0

    # <-- Properties -->
    @property
    def snakes_dead(self) -> torch.Tensor:
        return torch.tensor(
            [not snake.alive for snake in self.snakes],
            dtype=torch.float
        )

    @property
    def rewards(self) -> torch.Tensor:
        return torch.tensor(
            [snake.reward for snake in self.snakes],
            dtype=torch.float)

    @property
    def snake_sizes(self) -> torch.Tensor:
        return torch.tensor(
            [snake.size for snake in self.snakes],
            dtype=torch.float
        )

    @property
    def snake_states(self) -> List[torch.Tensor]:
        return [self.get_state(snake) for snake in self.snakes]

    @property
    def snake_state_size(self) -> int:
        return len(self.snake_states[0]) if self.snakes else 0

    @property
    def game_state(self) -> List[List[str]]:
        return self.map


if __name__ == "__main__":
    from src.config.settings import config

    env = Environment(config)
    print(env.snakes_dead)
    print(env.rewards)
    print(env.snake_sizes)
    print(env.snake_states)

    # take a step
    env.process_human_turn()

    print(env.snakes_dead)
    print(env.rewards)
    print(env.snake_sizes)
    print(env.snake_states)
