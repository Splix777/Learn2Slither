from typing import List, Optional
import pygame
from pygame.event import Event
from time import time

from src.config.settings import Config
from src.ui.pygame.screens.base_screen import BaseScreen

class LandingScreen(BaseScreen):
    def __init__(self, config: Config, theme: str) -> None:
        self.config: Config = config
        self.theme: str = theme

        self.large_font = pygame.font.Font(None, 120)
        self.small_font = pygame.font.Font(None, 24)

        self.title_text = self.large_font.render(
            "Battle Snakes", True, (18, 90, 60)
        )
        self.shadow_text = self.large_font.render(
            "Battle Snakes", True, (0, 0, 0)
        )
        self.sub_text = self.small_font.render(
            "Press ENTER or SPACE to Start", True, (200, 200, 200)
        )

        self.blink_start_time: float = time()
        self.blink_duration = 0.5
        self.show_instructions = True
        self.next_sreen = None

    def handle_input(self, events: List[Event]):
        """Handle input to navigate to the home screen."""
        if next(
            (
                event
                for event in events
                if event.type == pygame.KEYDOWN
                and event.key in (pygame.K_RETURN, pygame.K_SPACE)
            ),
            None,
        ):
            self.next_sreen = "home"

    def update(self) -> None:
        """Update the landing screen."""
        current_time: float = time()
        if current_time - self.blink_start_time >= self.blink_duration:
            self.show_instructions = not self.show_instructions
            self.blink_start_time = current_time

    def render(self, screen: pygame.Surface) -> None:
        """Render the landing screen."""     
        screen.fill((30, 30, 30))

        background: pygame.Surface = pygame.image.load(
            self.config.textures.backgrounds.dark
            if self.theme == "dark"
            else self.config.textures.backgrounds.light
        )
        screen.blit(
            pygame.transform.scale(background, screen.get_size()), (0, 0)
        )

        title_x = screen.get_width() // 2 - self.title_text.get_width() // 2
        title_y = screen.get_height() // 2 - self.title_text.get_height() // 2
        screen.blit(self.shadow_text, (title_x + 5, title_y + 5))
        screen.blit(self.title_text, (title_x, title_y))

        if self.show_instructions:
            inst_x = screen.get_width() // 2 - self.sub_text.get_width() // 2
            inst_y = title_y + self.title_text.get_height() + 20
            screen.blit(self.sub_text, (inst_x, inst_y))

    def get_next_screen(self) -> Optional[str]:
        """Return the next screen if a transition is needed."""
        return self.next_sreen