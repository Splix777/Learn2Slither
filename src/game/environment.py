import random
from typing import List, Tuple, Dict
from itertools import product
import torch

from src.game.snake import Snake
from src.config.settings import Config
from src.game.models.directions import Direction


class Environment:
    def __init__(self, config: Config, snakes: List[Snake]):
        self.config: Config = config
        # <-- Map Dimensions -->
        self.height: int = config.map.board_size.height
        self.width: int = config.map.board_size.width
        # <-- Apples -->
        self.red_apples: List[Tuple[int, int]] = []
        self.green_apples: List[Tuple[int, int]] = []
        self.max_red_apples: int = config.map.red_apples
        self.max_green_apples: int = config.map.green_apples
        # <-- Snakes -->
        self.num_snakes: int = len(snakes)
        # <-- Initialize Map -->
        self.snakes: List[Snake] = snakes
        self.map: List[List[str]] = []
        self.reset()

    def __repr__(self) -> str:
        return f"Enviroment - Height: {self.height} - Width: {self.width}"

    def reset(self) -> None:
        """Reset the game environment."""
        starting_pos = self._starting_positions()
        for index, (dir, pos) in enumerate(starting_pos.items()):
            self.snakes[index].reset(pos, dir)
        self.green_apples = []
        self.red_apples = []
        self.draw_map()

    # <-- Map Creation and Rendering -->
    def draw_map(self) -> None:
        """Create a map with the given configuration."""
        rendered_map: List[List[str]] = self._create_blank_map()
        self._render_snakes(empty_map)
        self._add_apples(empty_map)
        self._render_apples(empty_map)
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

    def _find_empty_cell(self, game_map: List[List[str]]) -> Tuple[int, int]:
        """Get a random empty position on the map."""
        while True:
            x: int = random.randint(1, self.width - 2)
            y: int = random.randint(1, self.height - 2)
            if game_map[y][x] == "empty":
                break
        return x, y

    def _render_snakes(self, game_map: List[List[str]]) -> None:
        """Place the snakes on the map."""
        for snake in self.snakes:
            if snake.alive:
                game_map[snake.head[0]][snake.head[1]] = "snake_head"
                for segment in snake.body[1:]:
                    game_map[segment[0]][segment[1]] = "snake_body"

    def _render_apples(self, game_map: List[List[str]]) -> None:
        """Place apples on the map."""
        for apple in self.green_apples:
            game_map[apple[0]][apple[1]] = "green_apple"
        for apple in self.red_apples:
            game_map[apple[0]][apple[1]] = "red_apple"

    def _add_apples(self, game_map: List[List[str]]) -> None:
        """Add apples to random empty spaces on the map."""
        while len(self.green_apples) < self.max_green_apples:
            x, y = self._find_empty_cell(game_map)
            self.green_apples.append((y, x))

        while len(self.red_apples) < self.max_red_apples:
            x, y = self._find_empty_cell(game_map)
            self.red_apples.append((y, x))
    
    def _starting_positions(self) -> Dict[Direction, Tuple[int, int]]:
        starting_positions: Dict[Direction, Tuple[int, int]] = {}
        starting_map = self._create_blank_map()
        for _ in range(self.num_snakes):
            while True:
                x, y = self._find_empty_cell(starting_map)
                direction = random.choice(list(Direction))
                if self._has_space((x, y), starting_map, direction):
                    starting_positions[direction] = (x, y)
                    break

        return starting_positions

    def _has_space(self, pos: Tuple[int, int], map: List[List[str]], direction: Direction) -> bool:
        x, y = pos
        dx, dy = direction.value
        total_steps = self.config.snake.start_size 
        ahead_steps = 2 

        for step in range(1, total_steps):
            nx, ny = x - dx * step, y - dy * step
            if (
                not 0 <= nx < self.height 
                or not 0 <= ny < self.width
                or map[nx][ny] != "empty"
            ):
                return False

        for step in range(1, ahead_steps):
            nx, ny = x + dx * step, y + dy * step
            if (
                not 0 <= nx < self.height
                or not 0 <= ny < self.width
                or map[nx][ny] != "empty"
            ):
                return False
            
        for step in range(1, total_steps):
            nx, ny = x - dx * step, y - dy * step
            map[nx][ny] = "reserved"
        for step in range(1, ahead_steps):
            nx, ny = x + dx * step, y + dy * step
            map[nx][ny] = "reserved"

        return True
    # < ------------------------------------->

    # <-- Events -->
    def check_collision(self, snake: Snake) -> str | None:
        """Check if the snake collided with a wall or another snake."""
        if self._collided_with_wall(snake):
            snake.delete()
            return "death"
        if self._collided_with_snake(snake):
            snake.delete()
            return "death"
        return None
    
    def check_apple_eaten(self, snake: Snake) -> str | None:
        """Check if the snake ate an apple."""
        if self.map[snake.head[0]][snake.head[1]] == "green_apple":
            self.green_apples.remove(snake.head)
            snake.green_apples_eaten += 1
            snake.steps_without_food = 0
            snake.grow()
            return "green_apple"
        elif self.map[snake.head[0]][snake.head[1]] == "red_apple":
            self.red_apples.remove(snake.head)
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

    def _collided_with_wall(self, snake: Snake) -> bool:
        """Checks if the snake collided with a wall."""
        return self.map[snake.head[0]][snake.head[1]] == "wall"

    def _collided_with_snake(self, snake: Snake) -> bool:
        """Checks if the snake collided with another snake or itself."""
        cell: str = self.map[snake.head[0]][snake.head[1]]
        if cell not in ["snake_body", "snake_head"]:
            return False
        for other_snake in self.snakes:
            if snake.head in other_snake.body and snake.id != other_snake.id:
                other_snake.kills += 1
        return True

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
    # < ------------------------------------->

    # <-- Game Steps -->
    def train_step(self):
        """Perform one step in the game for all snakes, ensuring proper next-step views."""
        buffer = {}
        for index, snake in enumerate(self.snakes):
            snake.reset_reward()
            state = self.get_state(snake)
            action = snake.move(state, learn=True)

            event = self._check_events(snake)
            if event:
                self._update_snake_rewards(snake, event)

            buffer[index] = (state, action, snake.reward, not snake.alive)

        next_states = [self.get_state(snake) for snake in self.snakes]

        for index, snake in enumerate(self.snakes):
            if snake.brain:
                state, action, reward, done = buffer[index]
                next_state = next_states[index]
                snake.brain.cache((
                    state,
                    action,
                    torch.tensor([reward], dtype=torch.float),
                    next_state,
                    torch.tensor([done], dtype=torch.float)
                ))

        self.draw_map()

    def _check_events(self, snake):
        events = [
            self.check_collision(snake),
            self.check_apple_eaten(snake),
            self.check_looping(snake),
        ]
        return next((event for event in events if event), None)

    def step(self) -> None:
        """Perform one step in the game based on the chosen action."""
        for snake in self.snakes:
            snake.move(self.get_state(snake))
            self.check_collision(snake)
            self.check_apple_eaten(snake)
            self.check_looping(snake)

        self.draw_map()
    # < ------------------------------------->

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
        if snake_id >= len(self.snakes):
            raise ValueError("Snake ID out of range.")
        return self.get_state(self.snakes[snake_id])

    def target_distances(self, x: int, y: int, target: str) -> torch.Tensor:
        """Calculate normalized distances to the closest target."""
        distances = torch.ones(4, dtype=torch.float)

        for index, direction in enumerate(Direction):
            dr, dc = direction.value
            for step in range(1, max(self.width, self.height)):
                nx, ny = x + dr * step, y + dc * step
                if not (0 <= nx < self.height and 0 <= ny < self.width):
                    distances[index] = self.normalize(
                        step - 1,
                        max(self.width, self.height)
                    )
                    break
                if self.map[nx][ny] == target:
                    distances[index] = self.normalize(
                        step,
                        max(self.width, self.height)
                    )
                    break

        return distances

    def assess_nearby_risks(self, x: int, y: int) -> torch.Tensor:
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

    def normalize(self, value: int, max_value: int) -> float:
        """Normalize a value between 0 and 1."""
        return value / max_value if max_value != 0 else 0

    def pretty_print_map(self) -> None:
        """Print the game map in a human-readable format."""
        if self.map:
            for row in self.map:
                print(" ".join([cell[:2] for cell in row]))
