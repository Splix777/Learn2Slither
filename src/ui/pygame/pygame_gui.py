"""PyGame GUI implementation."""

import os
from typing import List, Optional

import pygame
from pygame.event import Event
from pygame.key import ScancodeWrapper

from src.ui.base_gui import GUI
from src.config.settings import Config
from src.ui.pygame.screens.game_screen import GameScreen
from src.ui.pygame.screens.home_screen import HomeScreen
from src.ui.pygame.screens.options_screen import OptionsScreen
from src.ui.utils.game_resource_loader import TextureLoader
from src.ui.pygame.screens.landing_screen import LandingScreen
from src.game.environment import Environment


class PygameGUI(GUI):
    def __init__(
        self, config: Config, size: Optional[tuple[int, int]] = None
    ) -> None:
        self.config: Config = config
        self.textures = TextureLoader(config)
        self.theme = config.visual.themes.selected_theme

        self.music_on = False
        pygame.init()
        if not self.is_running_in_docker():
            pygame.mixer.init()
            self.music_on = True
            self.load_background_music("home")

        self.current_resolution = size or (1360, 960)
        self.screen = pygame.display.set_mode(self.current_resolution)
        self.clock = pygame.time.Clock()

        self.screens = {
            "landing": lambda: LandingScreen(config, self.theme),
            "home": lambda: HomeScreen(config, self.theme),
            "human_vs_ai": lambda: GameScreen(
                config, self.theme, "human_vs_ai"
            ),
            "ai_solo": lambda: GameScreen(config, self.theme, "ai_solo"),
            "options": lambda: OptionsScreen(config, self.theme, self),
        }
        self.current_screen = self.screens["landing"]()

        pygame.display.set_caption("Battle Snakes")
        icon = pygame.image.load(config.textures.green_apple.dark)
        pygame.display.set_icon(icon)
        self.running = True

    def run(self) -> None:
        while self.running:
            events = pygame.event.get()
            self.handle_global_events(events)

            if next_screen_name := self.current_screen.get_next_screen():
                self.transition_to_screen(next_screen_name)

            self.current_screen.handle_input(events)
            self.current_screen.update()

            self.current_screen.render(self.screen)
            self.render_fps(self.screen)

            pygame.display.flip()
            self.clock.tick(5)

    def transition_to_screen(self, screen_name) -> None:
        if screen_name not in self.screens:
            raise ValueError(f"Unknown screen: {screen_name}")
        self.current_screen = self.screens[screen_name]()
        if not self.is_running_in_docker():
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

            keys: ScancodeWrapper = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                if isinstance(self.current_screen, GameScreen):
                    self.current_screen.return_home()
                else:
                    self.running = False
                    pygame.quit()

    def resize_screen(self, new_size: tuple[int, int]) -> None:
        """Resize the screen."""
        self.current_resolution = new_size
        self.screen = pygame.display.set_mode(
            self.current_resolution, pygame.RESIZABLE
        )

    def load_background_music(self, track: str) -> None:
        """Load the background music."""
        if self.music_on:
            pygame.mixer.music.load(
                str(self.config.pygame_audio.home)
                if track == "home"
                else str(self.config.pygame_audio.game)
            )
            pygame.mixer.music.play(-1)
            self.set_music_volume(0.5)

    def toggle_music(self) -> None:
        """Toggle the music on or off."""
        self.music_on = not self.music_on
        if self.music_on:
            pygame.mixer.music.unpause()
        else:
            pygame.mixer.music.pause()

    def set_music_volume(self, volume: float) -> None:
        """Set the volume of the background music."""
        pygame.mixer.music.set_volume(volume)

    def render_map(self, env: Environment) -> None:
        """Render the game environment with the map centered."""
        if not pygame.display.get_active():
            return

        events = pygame.event.get()
        self.handle_global_events(events)

        if not self.screen:
            return

        self.screen.fill(
            color=(18, 90, 60),
        )

        board_width = env.width * self.textures.texture_size
        board_height = env.height * self.textures.texture_size

        centered_x = (self.current_resolution[0] - board_width) // 2
        centered_y = (self.current_resolution[1] - board_height) // 2

        for row_num, row in enumerate(env.map):
            for col_num, cell in enumerate(row):
                texture: pygame.Surface = pygame.image.load(
                    str(self.textures.textures[cell])
                )
                self.screen.blit(
                    texture,
                    (
                        centered_x + col_num * self.textures.texture_size,
                        centered_y + row_num * self.textures.texture_size,
                    ),
                )

        pygame.display.update()

    def is_running_in_docker(self) -> bool:
        return os.path.exists("/.dockerenv")
