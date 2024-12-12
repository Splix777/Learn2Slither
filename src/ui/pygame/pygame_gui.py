from typing import List

import pygame
from pygame.event import Event
from pygame.key import ScancodeWrapper

from src.ui.base_gui import GUI
from src.config.settings import Config
from src.game.environment import Environment
from src.ui.pygame.screens.game_screen import GameScreen
from src.ui.pygame.screens.home_screen import HomeScreen
from src.ui.pygame.screens.landing_screen import LandingScreen


class PygameGUI(GUI):
    def __init__(self, config: Config, environment: Environment):
        self.config: Config = config
        self.environment: Environment = environment

        # Initialize Pygame and set up screens
        pygame.init()
        self.base_resolution = (640, 480)
        self.current_resolution = self.base_resolution
        self.screen = pygame.display.set_mode(
            self.base_resolution, pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()

        self.screens = {
            "landing": LandingScreen(config),
            "home": HomeScreen(config),
            "game": GameScreen(config),
        }
        self.current_screen_name = "landing"
        self.running = True

        # Set window title and icon
        pygame.display.set_caption("Battle Snakes")
        icon = pygame.image.load(config.pygame_textures.green_apple.dark)
        pygame.display.set_icon(icon)

    def run(self) -> None:
        """Main game loop."""
        while self.running:
            events: List[Event] = pygame.event.get()

            # Handle global events
            self.handle_global_events(events)

            # Update environment if on the game screen
            if isinstance(self.screens["game"], GameScreen):
                self.screens["game"].update_map_state(
                    self.environment.game_state
                )

            # Handle input and switch screens if needed
            current_screen = self.screens[self.current_screen_name]
            if next_screen_name := current_screen.handle_input(events):
                self.current_screen_name: str = next_screen_name

            # Render the current screen
            current_screen.update()
            current_screen.render(self.screen)

            # Render the FPS counter
            self.render_fps(self.screen)

            pygame.display.flip()
            self.clock.tick(60)

    def render_fps(self, screen: pygame.Surface) -> None:
        """Render the FPS counter in the top-right corner."""
        font = pygame.font.Font(None, 12)
        fps: float = self.clock.get_fps()
        fps_text: pygame.Surface = font.render(
            f"FPS: {int(fps)}", True, (255, 255, 255)
        )
        screen.blit(
            fps_text,
            (self.current_resolution[0] - fps_text.get_width() - 10, 10),
        )

    def handle_global_events(self, events: List[Event]) -> None:
        """Handle events that apply globally across all screens."""
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False

            # Handle global keyboard events
            keys: ScancodeWrapper = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                self.running = False


if __name__ == "__main__":
    from src.game.environment import Environment
    from src.config.settings import config

    config: Config
    environment = Environment(config)
    gui = PygameGUI(config, environment)
    gui.run()
