from typing import List

import pygame
from pygame.event import Event
from pygame.key import ScancodeWrapper

from src.ui.base_gui import GUI
from src.config.settings import Config
from src.ui.pygame.screens.game_screen import GameScreen
from src.ui.pygame.screens.home_screen import HomeScreen
from src.ui.pygame.screens.landing_screen import LandingScreen
from src.ui.utils.texture_loader import TextureLoader


class PygameGUI(GUI):
    def __init__(self, config: Config):
        self.config: Config = config
        self.textures = TextureLoader(config)

        pygame.init()
        self.base_resolution = (
            config.map.board_size.width * self.textures.texture_size,
            config.map.board_size.height * self.textures.texture_size,
        )
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

            # Update the current screen
            current_screen = self.screens[self.current_screen_name]
            if next_screen_name := current_screen.handle_input(events):
                if next_screen_name in ["human_vs_ai", "ai_solo"]:
                    self.screens["game"] = GameScreen(
                        self.config, next_screen_name
                    )
                    self.current_screen_name = "game"
                else:
                    self.current_screen_name = next_screen_name

            # Check for resize request
            if new_size := current_screen.get_requested_screen_size():
                self.resize_screen(new_size)

            # Render the current screen
            current_screen.update()
            current_screen.render(self.screen)

            # Render the FPS counter
            self.render_fps(self.screen)

            pygame.display.flip()
            self.clock.tick(5)

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

    def resize_screen(self, new_size: tuple[int, int]) -> None:
        """Resize the screen."""
        self.current_resolution = new_size
        self.screen = pygame.display.set_mode(
            self.current_resolution, pygame.RESIZABLE
        )

    def training_render(self, game_state: List[List[str]]) -> None:
        """Render the game environment."""
        if not self.screen:
            return
        self.screen.fill(color=(18, 90, 60))
        for row_num, row in enumerate(iterable=game_state):
            for col_num, cell in enumerate(iterable=row):
                texture: pygame.Surface = pygame.image.load(
                    str(self.textures.textures[cell])
                )
                self.screen.blit(
                    texture,
                    (
                        col_num * self.textures.texture_size,
                        row_num * self.textures.texture_size,
                    ),
                )

        pygame.display.update()


if __name__ == "__main__":
    from src.config.settings import config

    gui = PygameGUI(config)
    gui.run()
