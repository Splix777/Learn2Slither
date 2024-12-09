"""Game manager module."""

from typing import List, Callable, Tuple, Optional, Dict

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
            if self.board.check_collision(snake=snake, snakes=self.board.snakes):
                snake.alive = False
            self.board.check_apple_eaten(snake)
            self.board.update_snake_position(snake)

        self.render()

    def step(self, action: list[int]) -> Tuple[float, bool, int]:
        """
        Perform one step in the game based on the chosen action,
        then return the new state, reward, and done flag.

        Args:
            action (int): The action chosen by the agent
                (0: up, 1: down, 2: left, 3: right).

        Returns:
            Tuple[List[int], float, bool]: The next state,
                the reward, and whether the game is done.
        """
        snake = self.board.snakes[0]
        snake.snake_controller(action)
        
        done = False
        reward = 0

        # self.update()
        self.board.add_apples()
        self.board.move_snake(snake=snake)
        if self.board.check_collision(snake=snake, snakes=self.board.snakes):
            snake.alive = False
            reward = self.config.rules.collisions.wall_collision.reward
            return reward, True, snake.size
        if apple := self.board.check_apple_eaten(snake):
            if apple == "green_apple":
                reward: int = self.config.rules.collisions.green_apple_collision.reward
            elif apple == "red_apple":
                reward: int = self.config.rules.collisions.red_apple_collision.reward
            if not snake.alive:
                reward: int = self.config.rules.collisions.snake_kill.reward
                return reward, True, snake.size
            
        self.board.update_snake_position(snake)

        self.render()

        # if not snake.alive:
        #     done = True
        #     reward: int = self.config.rules.collisions.wall_collision.reward
        #     return reward, done, snake.size

        # for _ in range(snake.green_apples_eaten):
        #     reward = self.config.rules.collisions.green_apple_collision.reward
        # for _ in range(snake.red_apples_eaten):
        #     reward = self.config.rules.collisions.red_apple_collision.reward
        # for _ in range(snake.kills):
        #     reward = self.config.rules.collisions.snake_kill.reward

        return reward, done, snake.size

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

    def get_distance_to_objects(self, head_x: int, head_y: int, target: str) -> List[int]:
        """
        Calculate the distance to the nearest target
        object in each cardinal direction.

        Args:
            head_x (int): Snake's head x-coordinate.
            head_y (int): Snake's head y-coordinate.
            target (str): Target object type (e.g., "green_apple").

        Returns:
            List[float]: Distances to the target object in
                [up, down, left, right].
        """
        directions = [0] * 4

        # Up
        for x in range(head_x - 1, -1, -1):
            if self.board.map[x][head_y] == target:
                directions[0] = 1
                break

        # Down
        for x in range(head_x + 1, self.board.height):
            if self.board.map[x][head_y] == target:
                directions[1] = 1
                break

        # Left
        for y in range(head_y - 1, -1, -1):
            if self.board.map[head_x][y] == target:
                directions[2] = 1
                break

        # Right
        for y in range(head_y + 1, self.board.width):
            if self.board.map[head_x][y] == target:
                directions[3] = 1
                break

        return directions

    def get_immediate_danger(self, head_x, head_y) -> List[int]:
        danger: List[int] = [0] * 4

        if head_x < 0 or self.board.map[head_x - 1][head_y] in ["wall", "snake_body"]:
            danger[0] = 1
        if head_x + 1 >= self.board.height or self.board.map[head_x + 1][head_y] in ["wall", "snake_body"]:
            danger[1] = 1
        if head_y < 0 or self.board.map[head_x][head_y - 1] in ["wall", "snake_body"]:
            danger[2] = 1
        if head_y + 1 >= self.board.width or self.board.map[head_x][head_y + 1] in ["wall", "snake_body"]:
            danger[3] = 1

        return danger

    def get_state(self, snake: Snake) -> List[int]:
        """
        Get an enhanced state representation for the snake.

        Args:
            snake (Snake): The snake to compute the state for.

        Returns:
            List[float]: The snake's state representation.
        """
        head_x, head_y = snake.head

        # Snake's current direction as one-hot encoding
        current_direction = snake.direction_one_hot

        # Distances to nearest green and red apples in each cardinal direction
        green_apple_distances = self.get_distance_to_objects(head_x, head_y, "green_apple")
        red_apple_distances = self.get_distance_to_objects(head_x, head_y, "red_apple")

        # Immediate danger flags
        immediate_danger = self.get_immediate_danger(head_x, head_y)

        # Possible direction based on current direction
        # 0: up, 1: down, 2: left, 3: right
        possible_directions = snake.possible_directions

        # Combine all features into a single state vector
        return (
            current_direction 
            + green_apple_distances
            + red_apple_distances 
            + immediate_danger 
            + possible_directions
        )

    def reset(self) -> None:
        self.board = GameBoard(config=self.config)



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