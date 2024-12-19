"""Game screen module."""

from typing import List, Optional, Callable
import pygame
from pygame.event import Event

from src.ai.agent import Agent
from src.config.settings import Config
from src.game.snake import Snake
from src.game.environment import Environment
from src.ui.pygame.screens.base_screen import BaseScreen
from src.ui.utils.game_resource_loader import TextureLoader
from src.ui.pygame.widgets.board_widget import BoardWidget
from src.ui.pygame.widgets.scores_widget import ScoreWidget


class GameScreen(BaseScreen):
    def __init__(
        self, config: Config, theme: str, mode: Optional[str] = None
    ) -> None:
        self.config = config
        self.theme = theme
        self.textures = TextureLoader(config)
        self.mode = mode
        self.current_resolution = (1360, 960)
        # Game state
        self.env: Optional[Environment] = None
        self.game_controller: Callable = lambda: None
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
            pygame.K_ESCAPE: lambda: self.return_home(),
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

        if self.env:
            self.env.step()

    def render(self, screen: pygame.Surface) -> None:
        """Render the current game state (environment)."""
        if not screen or not self.env:
            return

        screen.fill(color=(0, 0, 0))
        self.board_widget.render(screen, game_board=self.env.map)
        if self.mode == "human_vs_ai":
            self.human_score_widget.render(
                screen, score=self.env.snakes[0].green_apples_eaten
            )
            self.ai_score_widget.render(
                screen, score=self.env.snakes[1].green_apples_eaten
            )
        elif self.mode == "ai_solo":
            self.ai_score_widget.render(
                screen, score=self.env.snakes[0].green_apples_eaten
            )

        pygame.display.update()

    def get_next_screen(self) -> Optional[str]:
        """Return the next screen if a transition is needed."""
        next_screen = self.next_screen
        self.next_screen = None
        return next_screen

    def return_home(self) -> None:
        """Return to the home screen."""
        if self.env:
            self.env = None
        self.next_screen = "home"

    def initialize_game(self) -> None:
        """Initialize the game based on the selected mode."""
        if self.mode == "human_vs_ai":
            self.start_human_vs_ai_game()
        elif self.mode == "ai_solo":
            self.start_ai_solo_game()

    def start_human_vs_ai_game(self) -> None:
        self._reconfigure(2)
        if self.env:
            self.game_controller = self.env.snakes[0].snake_controller
        self.human_score_widget = ScoreWidget(
            pos=(10, 10),
            size=(200, 100),
            z=1,
            textures=self.textures,
            snake_id=1,
        )
        self.ai_score_widget = ScoreWidget(
            pos=(10, 120),
            size=(200, 100),
            z=1,
            textures=self.textures,
            snake_id=2,
        )

    def start_ai_solo_game(self) -> None:
        self._reconfigure(1)
        self.ai_score_widget = ScoreWidget(
            pos=(10, 10),
            size=(200, 100),
            z=1,
            textures=self.textures,
            snake_id=1,
        )

    def _reconfigure(self, snakes: int) -> None:
        self.pause = False
        self.exit = False
        brain = Agent(
            config=self.config, path=self.config.snake.selected_difficulty
        )
        ai_snake = Snake(1, self.config, brain)
        if snakes == 2:
            human_snake = Snake(0, self.config)
            self.env = Environment(self.config, [human_snake, ai_snake])
        else:
            self.env = Environment(self.config, [ai_snake])

        board_width = self.env.width * self.textures.texture_size
        board_height = self.env.height * self.textures.texture_size

        centered_x = (self.current_resolution[0] - board_width) // 2
        centered_y = (self.current_resolution[1] - board_height) // 2

        self.board_widget = BoardWidget(
            pos=(centered_x, centered_y),
            size=(self.textures.texture_size, self.textures.texture_size),
            z=0,
            textures=self.textures,
        )
