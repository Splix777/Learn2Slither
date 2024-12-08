"""Game manager module."""

import os
from pathlib import Path
from typing import List, Callable, Tuple

import pygame

from rich.text import Text
from rich.console import Console

from src.config.settings import Config, config
from src.game.board import GameBoard
from src.game.snake import Snake
from src.ui.game_textures import GameTextures
from src.utils.keyboard_handler import KeyListener


class GameManager:
    def __init__(self, config: Config, textures: GameTextures) -> None:
        """
        Game manager class.

        Args:
            config (Config): The game configuration.
            textures (GameTextures): The game textures.
        """
        self.config: Config = config
        self.textures: GameTextures = textures
        self.board: GameBoard = GameBoard(config=config)
        self.game_controllers: dict[int, Callable] = {}
        self.game_visuals: str = self.config.visual.modes.mode
        self.initialize()

    def initialize(self) -> None:
        for snake in self.board.snakes:
            self.game_controllers[snake.id] = snake.change_direction

        if self.game_visuals == "cli":
            self.console: Console = Console()
            self.render: Callable = self.render_ascii
        else:
            pygame.init()
            self.window: pygame.Surface = pygame.display.set_mode(
                size=(self.board.width * 20, self.board.height * 20)
            )
            self.render: Callable = self.render_pygame

    def update(self) -> None:
        self.board.add_apples()
        for snake in self.board.snakes:
            self.board.move_snake(snake=snake)
            if self.board.check_collision(snake=snake, snakes=self.board.snakes):
                snake.alive = False
            self.board.check_apple_eaten(snake)
            self.board.update_snake_position(snake)

        self.render()

    def render_ascii(self) -> None:
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
        self.window.fill(color=(0, 0, 0))
        for row_num, row in enumerate(iterable=self.board.map):
            for col_num, cell in enumerate(iterable=row):
                texture: pygame.Surface = pygame.image.load(
                    filename=str(self.textures.textures[cell])
                )
                self.window.blit(texture, (col_num * 20, row_num * 20))

        score_positions = [
            (0, 0),
            (self.window.get_width() - 220, 0),
            (0, self.window.get_height() - 18),
            (self.window.get_width() - 220, self.window.get_height() - 18),
        ]

        font = pygame.font.Font(name=None, size=24)
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

    def get_snake_vision(self, snake: Snake) -> List[int]:
        """
        Get the snake's vision of the board as a flattened 
        list of integers. The vision is a 1D array of the
        board elements in the following order: up, down, left, right.
        The snake can see in the four cardinal directions up to the
        walls of the board.

        Args:
            snake (Snake): The snake to compute the vision for.

        Returns:
            List[int]: The snake's vision of the board.
        """
        head_x, head_y = snake.head
        board_height = self.config.map.board_size.height
        board_width = self.config.map.board_size.width

        # Mapping of board elements to numeric values
        state_mapping = {
            "wall": 0,
            "snake_head": 1,
            "snake_body": 2,
            "green_apple": 3,
            "red_apple": 4,
            "empty": 5,
        }

        # Compute vision in four directions and convert to numeric values
        vision = {
            "up": [
                state_mapping[self.board.map[x][head_y]]
                for x in range(head_x - 1, -1, -1)
            ],
            "down": [
                state_mapping[self.board.map[x][head_y]]
                for x in range(head_x + 1, board_height)
            ],
            "left": [
                state_mapping[self.board.map[head_x][y]]
                for y in range(head_y - 1, -1, -1)
            ],
            "right": [
                state_mapping[self.board.map[head_x][y]]
                for y in range(head_y + 1, board_width)
            ],
        }

        # Flatten the vision (concatenate the four directions) and return
        return vision["up"] + vision["down"] + vision["left"] + vision["right"]

    def step(self, action: int) -> Tuple[List[int], float, bool]:
        """
        Perform one step in the game based on the chosen action, then return the new state, reward, and done flag.

        Args:
            action (int): The action chosen by the agent (0: up, 1: down, 2: left, 3: right).

        Returns:
            Tuple[List[int], float, bool]: The next state, the reward, and whether the game is done.
        """
        snake = self.snakes[0]  # Assuming the agent is controlling the first snake

        # Get the current state (vision of the board)
        state = self.get_snake_vision(snake)

        # Apply the action
        snake.change_direction(action)

        # Move the snake and update the game state
        self.update()

        # Check for collisions and apply rewards
        done = False
        reward = 0

        # Check if the snake collided with a wall or itself
        if self.board.check_collision(snake, self.snakes):
            done = True
            reward = -10  # Negative reward for hitting a wall or itself

        # Check if the snake ate an apple
        elif self.board.map[snake.head[0]][snake.head[1]] == "green_apple":
            reward = 10  # Positive reward for eating a green apple
        elif self.board.map[snake.head[0]][snake.head[1]] == "red_apple":
            reward = 5  # Positive reward for eating a red apple

        # Return the next state, reward, and whether the game is done
        next_state = self.get_snake_vision(snake)
        return next_state, reward, done



if __name__ == "__main__":
    import time
    fps = 6

    textures = GameTextures(config)
    game_manager = GameManager(config, textures)
    key_listener = KeyListener()
    while True:
        if key := key_listener.get_key():
            game_manager.game_controllers[0](int(key))
        game_manager.update()
        time.sleep(1 / fps)
        if not any(snake.alive for snake in game_manager.board.snakes):
            break
    key_listener.listener.stop()