from typing import List, Optional, Callable
import pygame
from pygame.event import Event

from src.ui.pygame.screens.base_screen import BaseScreen
from src.config.settings import Config
from src.game.environment import Environment
from src.game.snake import Snake
from src.ui.utils.game_resource_loader import TextureLoader
from src.ai.agent import Agent


class GameScreen(BaseScreen):
    def __init__(
        self, config: Config, theme: str, mode: Optional[str] = None
    ) -> None:
        self.config = config
        self.theme = theme
        self.textures = TextureLoader(config)
        self.mode = mode
        self.current_resolution = (1260, 960)
        # Game state
        self.env: Optional[Environment] = None
        self.game_controller: Callable = lambda: None
        self.ai_controller: Callable = lambda: None
        self.bindings: dict[str, int] = {"w": 0, "s": 1, "a": 2, "d": 3}
        self.pause = False
        self.exit = False
        # Text
        self.large_font = pygame.font.Font(None, 120)
        self.game_over_text = self.large_font.render(
            "Game Over", True, (255, 0, 0)
        )
        self.next_screen: Optional[str] = None
        self.initialize_game()

    def handle_input(self, events: List[Event]):
        """Handle player input specific to the game."""
        key_mapping = {
            pygame.K_ESCAPE: lambda: "home",
            pygame.K_SPACE: lambda: setattr(self, "pause", not self.pause),
            pygame.K_w: lambda: self.game_controller(self.bindings["w"]),
            pygame.K_s: lambda: self.game_controller(self.bindings["s"]),
            pygame.K_a: lambda: self.game_controller(self.bindings["a"]),
            pygame.K_d: lambda: self.game_controller(self.bindings["d"]),
        }
        for event in events:
            if event.type == pygame.KEYDOWN and event.key in key_mapping:
                action = key_mapping[event.key]
                if result := action():
                    if result == "home":
                        self.next_screen = str(result)

    def update(self) -> None:
        """Update game logic."""
        if self.pause or self.exit:
            return
        self._handle_ai_movement()

        if self.env and self.env.map:
            self.game_state = self.env.map

    def _handle_ai_movement(self) -> None:
        """Handle AI movement based on the current state."""
        if self.env:
            self.env.step()

    def render(self, screen: pygame.Surface) -> None:
        """Render the current game state (environment)."""
        if not screen or not self.env:
            return

        screen.fill(color=(18, 90, 60))

        map_height = len(self.env.map)
        map_width = len(self.env.map[0]) if map_height > 0 else 0

        x_offset = (
            self.current_resolution[0]
            - map_width * self.textures.texture_size
        ) // 2
        y_offset = (
            self.current_resolution[1]
            - map_height * self.textures.texture_size
        ) // 2

        for row_num, row in enumerate(self.env.map):
            for col_num, cell in enumerate(row):
                texture: pygame.Surface = pygame.image.load(
                    str(self.textures.textures[cell])
                )
                screen.blit(
                    texture,
                    (
                        x_offset + col_num * self.textures.texture_size,
                        y_offset + row_num * self.textures.texture_size,
                    ),
                )

        font = pygame.font.Font(None, 20)
        for i, snake in enumerate(self.env.snakes):
            text_surface = font.render(
                f"Snake {i+1} Apples: {snake.green_apples_eaten}",
                True,
                (255, 255, 255),
            )
            screen.blit(text_surface, (10, 10 + i * 30))

        pygame.display.update()

    def get_next_screen(self) -> Optional[str]:
        """Return the next screen if a transition is needed."""
        return self.next_screen

    def initialize_game(self) -> None:
        """Initialize the game based on the selected mode."""
        if self.mode == "human_vs_ai":
            self.start_human_vs_ai_game()
        elif self.mode == "ai_solo":
            self.start_ai_solo_game()

    def preload_textures(self) -> None:
        """Preload textures into memory for access during rendering."""
        self.texture_cache = {
            cell_type: pygame.image.load(
                str(self.textures.textures[cell_type])
            )
            for cell_type in self.textures.textures
        }

    def start_human_vs_ai_game(self) -> None:
        self._reconfigure(2)
        if self.env:
            self.game_controller = self.env.snakes[0].snake_controller

    def start_ai_solo_game(self) -> None:
        self._reconfigure(1)

    def _reconfigure(self, snakes: int) -> None:
        self.pause = False
        self.exit = False
        brain = Agent(
            config=self.config, path=self.config.snake.difficulty.ai_hard
        )
        ai_snake = Snake(1, self.config, brain)
        if snakes == 2:
            human_snake = Snake(0, self.config)
            self.env = Environment(self.config, [human_snake, ai_snake])
        else:
            self.env = Environment(self.config, [ai_snake])
