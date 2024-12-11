"""Game manager module."""

from typing import Callable, Optional, Dict

import pygame

from rich.text import Text
from rich.console import Console

from src.config.settings import Config
from src.game.enviroment import Enviroment
from src.ui.texture_loader import TextureLoader


class GUI:
    def __init__(
        self, config: Config, textures: Optional[TextureLoader] = None
    ) -> None:
        """
        Game manager class.

        Args:
            config (Config): The game configuration.
            textures (GameTextures): The game textures.
        """
        self.config: Config = config
        self.game_controllers: Dict[int, Callable[[list[int]], None]] = {}
        # <-- Visual Settings -->
        self.textures: TextureLoader = textures or TextureLoader(config)
        self.texture_size: int = config.pygame_textures.texture_size
        self.game_visuals: str = self.config.visual.modes.mode
        # <-- Game Data -->
        self.board_width: int = self.config.map.board_size.width
        self.board_height: int = self.config.map.board_size.height
        # <-- GUI Optionals -->
        self.console: Optional[Console] = None
        self.window: Optional[pygame.Surface] = None
        self.render: Callable = lambda: None
        # <-- Initialize -->
        self.initialize()

    def initialize(self) -> None:
        if self.game_visuals == "cli":
            self.console = Console()
            self.render: Callable = self.render_ascii
        else:
            pygame.init()
            self.window = pygame.display.set_mode(
                size=(
                    self.board_width * self.texture_size,
                    self.board_height * self.texture_size,
                )
            )
            self.render: Callable = self.render_pygame

    def render_ascii(self, enviroment: Enviroment) -> None:
        if not self.console:
            print("Console not initialized.")
            return
        board_text = Text()
        self.console.clear()
        for row in enviroment.map:
            for cell in row:
                board_text += self.textures.textures[cell]
            board_text += "\n"
        self.console.print(board_text, justify="center")
        for snake in enviroment.snakes:
            self.console.print(
                f"Snake {snake.id + 1}: "
                f"Length: {len(snake.body)} | Kills: {snake.kills}",
                justify="center",
            )

    def render_pygame(self, enviroment: Enviroment) -> None:
        if not self.window:
            print("Window not initialized.")
            return
        self.window.fill(color=(18, 90, 60))
        for row_num, row in enumerate(iterable=enviroment.map):
            for col_num, cell in enumerate(iterable=row):
                texture: pygame.Surface = pygame.image.load(
                    str(self.textures.textures[cell])
                )
                self.window.blit(
                    texture,
                    (
                        col_num * self.texture_size,
                        row_num * self.texture_size,
                    ),
                )

        score_positions = [
            (0, 0),
            (self.window.get_width() - 220, 0),
            (0, self.window.get_height() - 18),
            (self.window.get_width() - 220, self.window.get_height() - 18),
        ]

        font = pygame.font.Font(None, size=24)
        for idx, snake in enumerate(iterable=enviroment.snakes):
            if idx < len(score_positions):
                text: pygame.Surface = font.render(
                    f"Snake {snake.id + 1}: "
                    f"Length: {len(snake.body)} | Kills: {snake.kills}",
                    True,
                    (255, 255, 255),
                )
                self.window.blit(text, score_positions[idx])

        pygame.display.update()
