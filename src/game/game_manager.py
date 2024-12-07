import os
import time
from typing import List

from src.config.settings import Config
from src.game.board import GameBoard
from src.game.snake import Snake
from src.ui.game_textures import GameTextures

class GameManager:
    def __init__(self, config: Config, textures: GameTextures):
        self.config: Config = config
        self.textures: GameTextures = textures
        self.board: GameBoard = GameBoard(config, textures)
        self.snakes: List[Snake] = self.create_snakes()

    def create_snakes(self) -> List[Snake]:
        snakes: List[Snake] = []
        for snake_num in range(self.config.map.snakes):
            snake = Snake(
                config=self.config,
                textures=self.textures,
                start_position=self.board.snake_starting_positions[snake_num],
                id=snake_num
            )
            snakes.append(snake)
            self.board.update_snake_position(snake)

        return snakes

    def start_game(self) -> None:
        self.board.add_apples()
        self.board.render()

        while True:
            os.system("clear")
            self.update()
            self.board.render()
            
            time.sleep(1)

    def update(self) -> None:
        for snake in self.snakes:
            snake.move()
            self.board.update_snake_position(snake)
            if self.board.check_collision(snake):
                snake.alive = False

        
if __name__ == "__main__":
    from src.config.settings import get_config

    config: Config = get_config()
    textures = GameTextures(config)
    game_manager = GameManager(config, textures)
    game_manager.start_game()