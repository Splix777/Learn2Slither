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
        # self.board.add_apples()

        if self.game_visuals == "cli":
            self.console = Console()
            self.render: Callable = self.render_ascii
        else:
            pygame.init()
            self.window = pygame.display.set_mode(
                size=(self.board.width * 32, self.board.height * 32)
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
        self.board.step(actions)

        self.render()

        return self.board.rewards, self.board.snakes_done, self.board.snakes_done

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
        self.window.fill(color=(18, 90, 60))
        for row_num, row in enumerate(iterable=self.board.map):
            for col_num, cell in enumerate(iterable=row):
                texture: pygame.Surface = pygame.image.load(
                    str(self.textures.textures[cell])
                )
                self.window.blit(texture, (col_num * 32, row_num * 32))

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

    def get_state_size(self) -> int:
        return self.board.snake_state_size

    def reset(self) -> None:
        self.board = Enviroment(config=self.config)
        for snake in self.board.snakes:
            self.game_controllers[snake.id] = snake.snake_controller