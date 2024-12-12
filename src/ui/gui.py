"""Game manager module."""

from typing import Callable, Optional, Dict
from abc import ABC, abstractmethod

import pygame
from rich.text import Text
from rich.console import Console

from src.config.settings import Config
from src.game.environment import Environment
from src.ui.texture_loader import TextureLoader


class GUI(ABC):
    """Abstract class for the game GUI."""

    @abstractmethod
    def run(self) -> None:
        """Run the GUI."""
        pass


    # def initialize(self) -> None:
    #     if self.game_visuals == "cli":
    #         self.console = Console()
    #         self.render: Callable = self.render_ascii
    #     else:
    #         pygame.init()
    #         self.window = pygame.display.set_mode(
    #             size=(
    #                 self.board_width * self.texture_size,
    #                 self.board_height * self.texture_size,
    #             )
    #         )
    #         self.render: Callable = self.render_pygame

    # def render_ascii(self, enviroment: Enviroment) -> None:
    #     if not self.console:
    #         print("Console not initialized.")
    #         return
    #     board_text = Text()
    #     self.console.clear()
    #     for row in enviroment.map:
    #         for cell in row:
    #             board_text += self.textures.textures[cell]
    #         board_text += "\n"
    #     self.console.print(board_text, justify="center")
    #     for snake in enviroment.snakes:
    #         self.console.print(
    #             f"Snake {snake.id + 1}: "
    #             f"Length: {len(snake.body)} | Kills: {snake.kills}",
    #             justify="center",
    #         )

    # def render_pygame(self, enviroment: Enviroment) -> None:
    #     if not self.window:
    #         print("Window not initialized.")
    #         return
    #     self.window.fill(color=(18, 90, 60))
    #     for row_num, row in enumerate(iterable=enviroment.map):
    #         for col_num, cell in enumerate(iterable=row):
    #             texture: pygame.Surface = pygame.image.load(
    #                 str(self.textures.textures[cell])
    #             )
    #             self.window.blit(
    #                 texture,
    #                 (
    #                     col_num * self.texture_size,
    #                     row_num * self.texture_size,
    #                 ),
    #             )

    #     score_positions = [
    #         (0, 0),
    #         (self.window.get_width() - 220, 0),
    #         (0, self.window.get_height() - 18),
    #         (self.window.get_width() - 220, self.window.get_height() - 18),
    #     ]

    #     font = pygame.font.Font(None, size=24)
    #     for idx, snake in enumerate(iterable=enviroment.snakes):
    #         if idx < len(score_positions):
    #             text: pygame.Surface = font.render(
    #                 f"Snake {snake.id + 1}: "
    #                 f"Length: {len(snake.body)} | Kills: {snake.kills}",
    #                 True,
    #                 (255, 255, 255),
    #             )
    #             self.window.blit(text, score_positions[idx])

    #     pygame.display.update()
