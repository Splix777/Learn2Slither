"""Game manager module."""

from typing import List, Callable, Tuple, Optional, Dict

import pygame

from rich.text import Text
from rich.console import Console

from src.config.settings import Config
from src.game.enviroment import Enviroment
from src.game.snake import Snake
from src.ui.texture_loader import TextureLoader
from src.utils.keyboard_handler import KeyListener


class Interpreter:
    def __init__(self, config: Config, textures: TextureLoader) -> None:
        """
        Game manager class.

        Args:
            config (Config): The game configuration.
            textures (GameTextures): The game textures.
        """
        self.config: Config = config
        self.textures: TextureLoader = textures
        self.board: Enviroment = Enviroment(config=config)
        self.game_visuals: str = self.config.visual.modes.mode
        self.game_controllers: Dict[int, Callable[[list[int]], None]] = {}
        # <-- GUI Optionals -->
        self.console: Optional[Console] = None
        self.window: Optional[pygame.Surface] = None
        self.render: Callable = lambda: None
        # <-- Initialize -->
        self.initialize()

    def initialize(self) -> None:
        for snake in self.board.snakes:
            self.game_controllers[snake.id] = snake.snake_controller
        self.board.add_apples()

        if self.game_visuals == "cli":
            self.console = Console()
            self.render: Callable = self.render_ascii
        else:
            pygame.init()
            self.window = pygame.display.set_mode(
                size=(self.board.width * 20, self.board.height * 20)
            )
            self.render: Callable = self.render_pygame

    def update(self) -> None:
        self.board.add_apples()
        for snake in self.board.snakes:
            self.board.move_snake(snake=snake)
            if self.board.check_collision(
                snake=snake, snakes=self.board.snakes
            ):
                snake.alive = False
            self.board.check_apple_eaten(snake)
            self.board.update_snake_position(snake)

        self.render()

    def step(self, actions: List[list[int]], snakes: List[Snake]):
        """
        Perform one step in the game based on the chosen action,
        then return the new state, reward, and done flag.

        Args:
            actions (List[list[int]]): Actions chosen by the agents
                (0: up, 1: down, 2: left, 3: right).
            snakes (List[Snake]): List of snake objects for the agents.

        Returns:
            Tuple[List[int], List[bool], List[int]]: The rewards, done flags, 
                and sizes of all snakes.
        """
        rewards: List[int] = [0] * len(snakes)
        dones: List[bool] = [False] * len(snakes)

        # Apply actions and update the board
        for i, (snake, action) in enumerate(zip(snakes, actions)):
            snake.snake_controller(action)  # Control snake based on action
            self.board.move_snake(snake)   # Move the snake on the board

            # Check for apple collisions
            apple = self.board.check_apple_eaten(snake)
            if apple == "green_apple":
                rewards[i] += self.config.rules.collisions.green_apple_collision.reward
            elif apple == "red_apple":
                rewards[i] += self.config.rules.collisions.red_apple_collision.reward

            # Check for wall or snake collisions
            if self.board.check_collision(snake, snakes):
                rewards[i] += self.config.rules.collisions.wall_collision.reward
                dones[i] = True  # Snake is done (e.g., collided)

        [self.board.update_snake_position(snake) for snake in snakes]

        # Add new apples to the board after all movements
        self.board.add_apples()

        # Render the updated board
        self.render()

        return rewards, dones, [snake.size for snake in snakes]

    def render_ascii(self) -> None:
        if not self.console:
            print("Console not initialized.")
            return
        board_text = Text()
        self.console.clear()
        for row in self.board.map:
            for cell in row:
                board_text += self.textures.textures[cell]
            board_text += "\n"
        self.console.print(board_text, justify="center")
        for snake in self.board.snakes:
            self.console.print(
                f"Snake {snake.id + 1}: "
                f"Length: {len(snake.body)} | Kills: {snake.kills}",
                justify="center",
            )

    def render_pygame(self) -> None:
        if not self.window:
            print("Window not initialized.")
            return
        self.window.fill(color=(0, 0, 0))
        for row_num, row in enumerate(iterable=self.board.map):
            for col_num, cell in enumerate(iterable=row):
                texture: pygame.Surface = pygame.image.load(
                    str(self.textures.textures[cell])
                )
                self.window.blit(texture, (col_num * 20, row_num * 20))

        score_positions = [
            (0, 0),
            (self.window.get_width() - 220, 0),
            (0, self.window.get_height() - 18),
            (self.window.get_width() - 220, self.window.get_height() - 18),
        ]

        font = pygame.font.Font(None, size=24)
        for idx, snake in enumerate(iterable=self.board.snakes):
            if idx < len(score_positions):
                text: pygame.Surface = font.render(
                    f"Snake {snake.id + 1}: "
                    f"Length: {len(snake.body)} | Kills: {snake.kills}",
                    True,
                    (255, 255, 255),
                )
                self.window.blit(text, score_positions[idx])

        pygame.display.update()

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
            if self.board.map[x - i][y] == target:
                directions[0] = i
                break

        # Down
        for i in range(1, self.board.height - x):
            if self.board.map[x + i][y] == target:
                directions[1] = i
                break

        # Left
        for i in range(1, y + 1):
            if self.board.map[x][y - i] == target:
                directions[2] = i
                break

        # Right
        for i in range(1, self.board.width - y):
            if self.board.map[x][y + i] == target:
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
        if x < 0 or self.board.map[x - 1][y] in collision:
            danger[0] = 1
        # Down
        if (
            x + 1 >= self.board.height
            or self.board.map[x + 1][y] in collision
        ):
            danger[1] = 1
        # Left
        if y < 0 or self.board.map[x][y - 1] in collision:
            danger[2] = 1
        # Right
        if y + 1 >= self.board.width or self.board.map[x][y + 1] in collision:
            danger[3] = 1

        return danger

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
        possible_directions = snake.possible_directions

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

    def get_state_size(self) -> int:
        return (
            len(self.get_state(self.board.snakes[0]))
            if self.board.snakes
            else 0
        )

    def reset(self) -> None:
        self.board = Enviroment(config=self.config)
