from typing import List, Optional, Callable
import pygame
from pygame.event import Event

import torch

from src.ui.pygame.screens.screen import BaseScreen
from src.config.settings import Config
from src.game.environment import Environment
from src.ui.utils.texture_loader import TextureLoader
from src.ai.agent import DeepQSnakeAgent


class GameScreen(BaseScreen):
    def __init__(self, config: Config, mode: Optional[str] = None) -> None:
        self.config = config
        self.textures = TextureLoader(config)
        self.mode = mode
        self.requested_screen_size = None
        # Game state
        self.env: Environment = Environment(self.config)
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
        # Initialize game state
        self.preload_textures()
        self.initialize_game()

    def handle_input(self, events: List[Event]) -> str:
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
                    print(result)
                    return result

        return ""

    def update(self) -> None:
        """Update game logic."""
        if self.pause or self.exit:
            return
        if self.mode == "human_vs_ai":
            self._handle_ai_movement(1)
        elif self.mode == "ai_solo":
            self._handle_ai_movement(0)

        if self.env and self.env.game_state:
            self.game_state = self.env.game_state

    def _handle_ai_movement(self, snake_index: int) -> None:
        """Handle AI movement based on the current state."""
        ai_actions = self.get_ai_movement(
            self.env.get_state_by_id(snake_index).tolist()
        )
        self.ai_controller(ai_actions)
        self.env.process_human_turn()

    def render(self, screen: pygame.Surface) -> None:
        """Render the current game state (environment)."""
        if not self.game_state:
            return
        screen.fill((255, 255, 255))

        for row_num, row in enumerate(self.game_state):
            for col_num, cell in enumerate(row):
                texture: pygame.Surface = self.texture_cache[cell]
                screen.blit(
                    texture,
                    (
                        col_num * self.textures.texture_size,
                        row_num * self.textures.texture_size,
                    ),
                )

        if not self.env.snakes[0].alive:
            screen.blit(
                self.game_over_text,
                (
                    screen.get_width() // 2
                    - self.game_over_text.get_width() // 2,
                    screen.get_height() // 2
                    - self.game_over_text.get_height() // 2,
                ),
            )

    def request_screen_resize(self, new_size: tuple[int, int]) -> None:
        """Request a screen resize to the specified dimensions."""
        if new_size[0] > 0 and new_size[1] > 0:
            self.requested_screen_size = new_size
        else:
            raise ValueError("Screen size must be positive dimensions.")

    def get_requested_screen_size(self) -> Optional[tuple[int, int]]:
        """Return the requested screen size and reset the request."""
        size = self.requested_screen_size
        self.requested_screen_size = None
        return size

    def initialize_game(self) -> None:
        """Initialize the game based on the selected mode."""
        if self.mode == "human_vs_ai":
            self.start_human_vs_ai_game()
        elif self.mode == "ai_solo":
            self.start_ai_solo_game()

    def preload_textures(self) -> None:
        """Preload textures into memory for faster access during rendering."""
        self.texture_cache = {
            cell_type: pygame.image.load(
                str(self.textures.textures[cell_type])
            )
            for cell_type in self.textures.textures
        }

    def start_human_vs_ai_game(self) -> None:
        self._reconfigure(2, (20, 20))
        self.game_controller = self.env.snakes[0].snake_controller
        self.ai_controller = self.env.snakes[1].snake_controller

    def start_ai_solo_game(self) -> None:
        self._reconfigure(1, (10, 10))
        self.game_controller = self.env.snakes[0].snake_controller

    def _reconfigure(self, snakes: int, dimensions: tuple[int, int]) -> None:
        self.config.map.snakes = snakes
        self.config.map.board_size.width = dimensions[0]
        self.config.map.board_size.height = dimensions[1]
        width = self.config.map.board_size.width * self.textures.texture_size
        height = (
            self.config.map.board_size.height * self.textures.texture_size
        )
        self.request_screen_resize((width, height))
        self.pause = False
        self.exit = False
        self.env = Environment(self.config)
        self.agent = DeepQSnakeAgent(
            input_size=self.env.snake_state_size,
            output_size=4,
            config=self.config,
        )
        ai_model_path = self.config.paths.models / "snake_best.pth"
        self.agent.load(ai_model_path)

    @torch.no_grad()
    def get_ai_movement(self, state: List[float]) -> List[int]:
        """Get the AI's movement based on the current state."""
        movement = [0] * 4
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        q_values = self.agent(state_tensor)
        move = int(torch.argmax(q_values).item())
        movement[move] = 1
        return movement
