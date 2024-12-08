import os
import time
from typing import List, Callable, Tuple
from pathlib import Path

import pygame

from rich.text import Text
from rich.console import Console

from src.config.settings import Config
from src.game.board import GameBoard
from src.game.snake import Snake
from src.ui.game_textures import GameTextures


class GameManager:
    def __init__(self, config: Config, textures: GameTextures):
        self.config: Config = config
        self.textures: GameTextures = textures
        self.board: GameBoard = GameBoard(config, textures)
        self.game_controllers: dict[int, Callable] = {}
        self.initialize()

    def initialize(self) -> None:
        self.snakes: List[Snake] = self.create_snakes()
        for snake in self.snakes:
            self.game_controllers[snake.id] = snake.change_direction

        self.game_visuals: str = self.config.visual.modes.mode
        if self.game_visuals == "cli":
            self.console = Console()
            self.render: Callable = self.ascii_mode()
        else:
            pygame.init()
            self.window: pygame.Surface = pygame.display.set_mode(
                (
                    self.config.map.board_size.width * 20,
                    self.config.map.board_size.height * 20,
                )
            )
            self.render: Callable = self.pygame_mode()

    def create_snakes(self) -> List[Snake]:
        snakes: List[Snake] = []
        for snake_num in range(self.config.map.snakes):
            snake = Snake(
                config=self.config,
                textures=self.textures,
                start_position=self.board.snake_starting_positions[snake_num],
                id=snake_num,
            )
            snakes.append(snake)
            self.board.update_snake_position(snake)

        return snakes

    def update(self) -> None:
        self.board.add_apples()
        for snake in self.snakes:
            self.board.move_snake(snake)
            if self.board.check_collision(snake, self.snakes):
                self.snakes.remove(snake)
                snake.alive = False
            self.board.check_apple_collision(snake)
            self.board.update_snake_position(snake)

        self.render()

    def ascii_mode(self) -> Callable:
        return self.render_ascii

    def pygame_mode(self) -> Callable:
        return self.render_pygame

    def render_ascii(self) -> None:
        board_text = Text()
        self.console.clear()
        for row in self.board.map:
            for cell in row:
                board_text += self.textures.textures[cell]
            board_text += "\n"
        self.console.print(board_text, justify="center")
        # Print snakes length and kills
        for snake in self.snakes:
            self.console.print(
                f"Snake {snake.id + 1}: "
                f"Length: {len(snake.body)} | Kills: {snake.kills}"
            )

    def render_pygame(self) -> None:
        self.window.fill((0, 0, 0))
        for row_num, row in enumerate(self.board.map):
            for col_num, cell in enumerate(row):
                if isinstance(self.textures.textures[cell], Path):
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

        # Render scores for each snake
        font = pygame.font.Font(None, 24)
        for idx, snake in enumerate(self.snakes):
            if idx < len(score_positions):
                text: pygame.Surface = font.render(
                    f"Snake {snake.id + 1}: "
                    f"Length: {len(snake.body)} | Kills: {snake.kills}",
                    True,
                    (255, 255, 255),
                )
                self.window.blit(text, score_positions[idx])

        # Update the display
        pygame.display.update()


if __name__ == "__main__":
    from src.config.settings import get_config
    from src.utils.keyboard_handler import KeyListener

    fps = 12

    config: Config = get_config()
    textures = GameTextures(config)
    game_manager = GameManager(config, textures)
    key_listener = KeyListener()
    while True:
        # Human can only be p1
        if key := key_listener.get_key():
            game_manager.game_controllers[0](key)
        game_manager.update()
        time.sleep(1 / fps)
