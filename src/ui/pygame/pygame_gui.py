from typing import List

import pygame
from pygame.event import Event
from pygame.key import ScancodeWrapper

from src.ui.base_gui import GUI
from src.config.settings import Config
from src.ui.pygame.screens.game_screen import GameScreen
from src.ui.pygame.screens.home_screen import HomeScreen
from src.ui.utils.game_resource_loader import TextureLoader
from src.ui.pygame.screens.landing_screen import LandingScreen
from src.game.environment import Environment


class PygameGUI(GUI):
    def __init__(self, config: Config):
        self.config = config
        self.textures = TextureLoader(config)
        self.theme = config.visual.themes.default_theme

        pygame.init()
        pygame.mixer.init()
        self.load_background_music("home")

        self.current_resolution = (1260, 960)
        self.screen = pygame.display.set_mode(self.current_resolution)
        self.clock = pygame.time.Clock()

        self.screens = {
            "landing": LandingScreen(config, self.theme),
            "home": HomeScreen(config, self.theme),
            "human_vs_ai": GameScreen(config, self.theme, "human_vs_ai"),
            "ai_solo": GameScreen(config, self.theme, "ai_solo"),
        }
        self.current_screen = self.screens["landing"]

        # Set window title and icon
        pygame.display.set_caption("Battle Snakes")
        icon = pygame.image.load(config.pygame_textures.green_apple.dark)
        pygame.display.set_icon(icon)
        self.running = True

    def run(self) -> None:
        while self.running:
            events = pygame.event.get()
            self.handle_global_events(events)

            # Get next screen or actions from current screen
            if next_screen_name := self.current_screen.get_next_screen():
                self.transition_to_screen(next_screen_name)

            # Handle screen-specific logic
            self.current_screen.handle_input(events)
            self.current_screen.update()

            # Render
            self.current_screen.render(self.screen)
            self.render_fps(self.screen)

            pygame.display.flip()
            self.clock.tick(5)

    def transition_to_screen(self, screen_name) -> None:
        if screen_name not in self.screens:
            raise ValueError(f"Unknown screen: {screen_name}")
        self.current_screen = self.screens[screen_name]
        if isinstance(self.current_screen, LandingScreen | HomeScreen):
            self.load_background_music("home")
        elif isinstance(self.current_screen, GameScreen):
            self.load_background_music("game")

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

    def load_background_music(self, track: str) -> None:
        """Load the background music."""
        pygame.mixer.music.load(
            str(self.config.pygame_audio.home)
            if track == "home"
            else str(self.config.pygame_audio.game)
        )
        pygame.mixer.music.play(-1)
        self.set_music_volume(0.5)

    def set_music_volume(self, volume: float) -> None:
        """Set the volume of the background music."""
        pygame.mixer.music.set_volume(volume)

    def render_map(self, env: Environment) -> None:
        """Render the game environment with the map centered."""
        if not self.screen:
            return
    
        self.screen.fill(color=(18, 90, 60))

        # Get map size
        map_height = len(env.map)
        map_width = len(env.map[0]) if map_height > 0 else 0

        x_offset = (
            self.current_resolution[0]
            - map_width * self.textures.texture_size
        ) // 2
        y_offset = (
            self.current_resolution[1]
            - map_height * self.textures.texture_size
        ) // 2

        # Render map
        for row_num, row in enumerate(env.map):
            for col_num, cell in enumerate(row):
                texture: pygame.Surface = pygame.image.load(
                    str(self.textures.textures[cell])
                )
                self.screen.blit(
                    texture,
                    (
                        x_offset + col_num * self.textures.texture_size,
                        y_offset + row_num * self.textures.texture_size,
                    ),
                )

        # Render game info on the left side
        font = pygame.font.Font(None, 20)
        for i, snake in enumerate(env.snakes):
            # Calculate position for each snake's apple count on the left side
            text_surface = font.render(
                f"Snake {i+1} Apples: {snake.green_apples_eaten}",
                True,
                (255, 255, 255),
            )
            self.screen.blit(text_surface, (10, 10 + i * 30))

        pygame.display.update()


if __name__ == "__main__":
    from src.config.settings import config

    gui = PygameGUI(config)
    gui.run()
