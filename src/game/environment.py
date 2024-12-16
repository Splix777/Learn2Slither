import random
from typing import List, Tuple, Dict, Set
from itertools import product
import torch

from src.game.snake import Snake
from src.config.settings import Config
from src.game.models.directions import Direction


class Environment:
    def __init__(self, config: Config, snakes: List[Snake]) -> None:
        self.config: Config = config
        # Map Dimensions
        self.height: int = config.map.board_size.height
        self.width: int = config.map.board_size.width
        # Apples
        self.red_apples: Set[Tuple[int, int]] = set()
        self.green_apples: Set[Tuple[int, int]] = set()
        self.max_red_apples: int = config.map.red_apples
        self.max_green_apples: int = config.map.green_apples
        # Snakes
        self.snakes: List[Snake] = snakes
        self.num_snakes: int = len(snakes)
        if self.num_snakes < 1:
            raise ValueError("At least one snake is required.")
        # <-- Initialize Map -->
        self.map: List[List[str]] = []
        self.reset()

    def __repr__(self) -> str:
        return f"Enviroment - Height: {self.height} - Width: {self.width}"

    # Reset Environment
    def reset(self) -> None:
        """Reset the game environment."""
        starting_pos = self._starting_positions()
        for index, (dir, x, y) in starting_pos.items():
            self.snakes[index].initialize((x, y), dir)
        self.green_apples = set()
        self.red_apples = set()
        self._draw_map()

    # Map Creation and Rendering
    def _starting_positions(self) -> Dict[int, Tuple[Direction, int, int]]:
        """Get the starting positions for all snakes."""
        starting_positions: Dict[int, Tuple[Direction, int, int]] = {}
        attempts = 10
        starting_map = self._create_blank_map()
        for i in range(self.num_snakes):
            while attempts:
                x, y = self._find_empty_cell(starting_map)
                direction = random.choice(list(Direction))
                if self._has_space((x, y), starting_map, direction):
                    starting_positions[i] = (direction, x, y)
                    break
                attempts -= 1

        if len(starting_positions) < self.num_snakes:
            raise ValueError("Couldn't find start positions for all snakes.")

        return starting_positions

    def _find_empty_cell(self, game_map: List[List[str]]) -> Tuple[int, int]:
        """Get a random empty position on the map."""
        while True:
            x: int = random.randint(1, self.width - 2)
            y: int = random.randint(1, self.height - 2)
            if game_map[y][x] == "empty":
                break
        return x, y

    def _has_space(
        self, pos: Tuple[int, int], map: List[List[str]], direction: Direction
    ) -> bool:
        """Check if there is enough space for the snake to start."""
        x, y = pos
        dx, dy = direction.value
        total_steps = self.config.snake.start_size
        ahead_steps = 3

        for steps, sign in [(total_steps, -1), (ahead_steps, 1)]:
            for step in range(1, steps):
                nx, ny = x + dx * step * sign, y + dy * step * sign
                if (
                    not 0 <= nx < self.height
                    or not 0 <= ny < self.width
                    or map[nx][ny] != "empty"
                ):
                    return False

        for steps, sign in [(total_steps, -1), (ahead_steps, 1)]:
            for step in range(1, steps):
                nx, ny = x + dx * step * sign, y + dy * step * sign
                map[x][y] = "reserved"
                map[nx][ny] = "reserved"

        return True

    def _draw_map(self) -> None:
        """Create a map with the given configuration."""
        rendered_map: List[List[str]] = self._create_blank_map()
        self._draw_snakes_on_map(rendered_map)
        self._populate_apples(rendered_map)
        self._draw_apples_on_map(rendered_map)
        self.map = rendered_map

    def _create_blank_map(self) -> List[List[str]]:
        """Create a blank map with the given configuration."""
        game_map = [["empty"] * self.width for _ in range(self.height)]

        for row, col in product(range(self.height), range(self.width)):
            if (
                row == 0
                or row == self.height - 1
                or col == 0
                or col == self.width - 1
            ):
                game_map[row][col] = "wall"

        return game_map

    def _draw_snakes_on_map(self, game_map: List[List[str]]) -> None:
        """Place the snakes on the map."""
        for snake in self.snakes:
            if snake.alive and snake.initialized:
                game_map[snake.head[0]][snake.head[1]] = "snake_head"
                for segment in snake.body[1:]:
                    game_map[segment[0]][segment[1]] = "snake_body"

    def _draw_apples_on_map(self, game_map: List[List[str]]) -> None:
        """Place apples on the map."""
        for apple in self.green_apples:
            if game_map[apple[0]][apple[1]] != "empty":
                continue
            game_map[apple[0]][apple[1]] = "green_apple"
        for apple in self.red_apples:
            if game_map[apple[0]][apple[1]] != "empty":
                continue
            game_map[apple[0]][apple[1]] = "red_apple"

    def _populate_apples(self, game_map: List[List[str]]) -> None:
        """Add apples to random empty spaces on the map."""
        while len(self.green_apples) < self.max_green_apples:
            x, y = self._find_empty_cell(game_map)
            self.green_apples.add((y, x))

        while len(self.red_apples) < self.max_red_apples:
            x, y = self._find_empty_cell(game_map)
            self.red_apples.add((y, x))

    # Events
    def _process_collision_events(self, snake) -> None:
        self._check_collision(snake)
        self._check_apple_eaten(snake)
        self._check_looping(snake)
    
    def _check_collision(self, snake: Snake) -> None:
        """Check if the snake collided with a wall or another snake."""
        if self._collided_wall(snake) or self._collided_snake(snake):
            snake.death()

    def _check_apple_eaten(self, snake: Snake) -> None:
        """Check if the snake ate an apple."""
        if self.map[snake.head[0]][snake.head[1]] == "green_apple":
            self.green_apples.remove(snake.head)
            snake.eat_green_apple()
        elif self.map[snake.head[0]][snake.head[1]] == "red_apple":
            self.red_apples.remove(snake.head)
            snake.eat_red_apple()

    def _check_looping(self, snake: Snake) -> None:
        """Check if the snake is looping."""
        if snake.steps_without_food >= self.config.rules.steps_no_apple:
            snake.looping()

    def _collided_wall(self, snake: Snake) -> bool:
        """Checks if the snake collided with a wall."""
        return self.map[snake.head[0]][snake.head[1]] == "wall"

    def _collided_snake(self, snake: Snake) -> bool:
        """Checks if the snake collided with another snake or itself."""
        cell: str = self.map[snake.head[0]][snake.head[1]]
        if cell not in ["snake_body", "snake_head"]:
            return False
        for other_snake in self.snakes:
            if snake.head in other_snake.body and snake.id != other_snake.id:
                other_snake.kill()
        return True

    # Game Steps
    def train_step(self) -> None:
        """Perform one step in the game for all snakes, ensuring proper next-step views."""
        buffer = {}
        for index, snake in enumerate(self.snakes):
            snake.reset_rewards()
            state = self.get_state(snake)
            action = snake.move(state, learn=True)
            self._process_collision_events(snake)

            buffer[index] = (state, action, snake.reward, not snake.alive)

        next_states = [self.get_state(snake) for snake in self.snakes]

        for index, snake in enumerate(self.snakes):
            if snake.brain:
                state, action, reward, done = buffer[index]
                next_state = next_states[index]
                snake.brain.cache(
                    (
                        state,
                        action,
                        torch.tensor([reward], dtype=torch.float),
                        next_state,
                        torch.tensor([done], dtype=torch.float),
                    )
                )

        self._draw_map()

    def step(self) -> None:
        """Perform one step in the game based on the chosen action."""
        for snake in self.snakes:
            snake.move(self.get_state(snake))
            self._process_collision_events(snake)

        self._draw_map()

    # State Representation
    def get_state(self, snake: Snake) -> torch.Tensor:
        """Get an optimized state representation for the snake."""
        head_x, head_y = snake.head

        return torch.cat(
            [
                torch.tensor(snake.one_hot_direction),
                torch.tensor(snake.one_hot_options),
                self._assess_nearby_risks(head_x, head_y),
                self._detect_surroundings(head_x, head_y, 40),
                self._segmented_vision(head_x, head_y, 40),
                self._evaluate_space_and_density(head_x, head_y),
            ]
        )

    def _assess_nearby_risks(self, x: int, y: int) -> torch.Tensor:
        """Detect immediate danger in the four cardinal directions."""
        danger = torch.zeros(4, dtype=torch.float)

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

    def _detect_surroundings(self, x: int, y: int, view: int) -> torch.Tensor:
        """Detect surrounding within a view range."""
        # 4 directions * 3 features (obstacle, green apple, red apple)
        result = torch.zeros(12, dtype=torch.float)
        max_lookahead = min(view, max(self.width, self.height))

        for index, direction in enumerate(Direction):
            dr, dc = direction.value
            for step in range(1, max_lookahead):
                nx, ny = x + dr * step, y + dc * step
                # 1. If out of bounds, set the obstacle flag
                if not (0 <= nx < self.height and 0 <= ny < self.width):
                    result[index * 3] = self._normalize(
                        step - 1, max(self.width, self.height)
                    )
                    break
                cell = self.map[nx][ny]
                # 2. If obstacle detected, set the obstacle flag
                if cell in ["wall", "snake_body", "snake_head"]:
                    result[index * 3] = self._normalize(
                        step, max(self.width, self.height)
                    )
                    break
                # 3. If apple detected, set the corresponding flag
                elif cell == "green_apple":
                    result[index * 3 + 1] = 1.0
                    if step > 1:
                        result[index * 3] = self._normalize(
                            step - 1, max(self.width, self.height)
                        )
                    break
                # 4. If red apple detected, set the corresponding flag
                elif cell == "red_apple":
                    result[index * 3 + 2] = 1.0
                    if step > 1:
                        result[index * 3] = self._normalize(
                            step - 1, max(self.width, self.height)
                        )
                    break
            else:
                result[index * 3] = 1.0

        return result

    def _segmented_vision(self, x: int, y: int, steps: int) -> torch.Tensor:
        """Provide segmented vision looking steps ahead in each direction."""
        # 4 directions * 4 features (wall, body, green apple, red apple)
        vision = torch.zeros(16, dtype=torch.float)
        directions = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])

        for index, (dx, dy) in enumerate(directions):
            for step in range(1, steps + 1):
                nx, ny = x + step * dx.item(), y + step * dy.item()
                cell = self.map[nx][ny]
                if nx < 0 or ny < 0 or nx >= self.height or ny >= self.width:
                    vision[index * 4] = 1.0
                    break
                elif cell == "wall":
                    vision[index * 4] = 1.0
                elif cell == "snake_body" or "snake_head":
                    vision[index * 4 + 1] = 1.0
                elif cell == "green_apple":
                    vision[index * 4 + 2] = 1.0
                elif cell == "red_apple":
                    vision[index * 4 + 3] = 1.0
                if cell != "empty":
                    break

        return vision

    def _evaluate_space_and_density(self, x: int, y: int) -> torch.Tensor:
        """Combine green and red apple density and open space ratio for each direction."""
        # 4 directions * (green apple, red apple, open space)
        result = torch.zeros(12, dtype=torch.float)
        directions = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])

        for index, (dx, dy) in enumerate(directions):
            green_apples = red_apples = open_space = 0
            for step in range(1, 6):
                nx, ny = x + step * dx.item(), y + step * dy.item()
                if nx < 0 or ny < 0 or nx >= self.height or ny >= self.width:
                    break
                cell = self.map[nx][ny]
                if cell == "green_apple":
                    green_apples += 1
                elif cell == "red_apple":
                    red_apples += 1
                elif cell == "empty":
                    open_space += 1

            result[index * 3] = green_apples / 5.0
            result[index * 3 + 1] = red_apples / 5.0
            result[index * 3 + 2] = open_space / 5.0

        return result

    def _normalize(self, value: int, max_value: int) -> float:
        """Normalize a value between 0 and 1."""
        return value / max_value if max_value != 0 else 0
