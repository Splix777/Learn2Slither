from typing import List
import pygame
from pygame.event import Event
from time import time

from src.ui.pygame.screens.screen import BaseScreen
from src.config.settings import Config


class LandingScreen(BaseScreen):
    def __init__(self, config: Config) -> None:
        self.config: Config = config

        # Fonts and colors
        self.large_font = pygame.font.Font(None, 120)
        self.small_font = pygame.font.Font(None, 24)
        self.title_color = (18, 90, 60)
        self.shadow_color = (0, 0, 0)
        self.instruction_color = (200, 200, 200)

        # Render title text with a shadow effect
        self.title_text = self.large_font.render(
            "Battle Snakes", True, self.title_color
        )
        self.shadow_text = self.large_font.render(
            "Battle Snakes", True, self.shadow_color
        )

        # Instruction text
        self.instructions_text = self.small_font.render(
            "Press ENTER or SPACE to Start", True, self.instruction_color
        )

        self.blink_start_time = time()
        self.blink_duration = 0.5
        self.show_instructions = True

    def handle_input(self, events: List[Event]) -> str:
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
            return "home"
        return ""

    def update(self) -> None:
        """Update the screen (e.g., animations)."""
        # Handle blinking effect for instruction text
        current_time: float = time()
        if current_time - self.blink_start_time >= self.blink_duration:
            self.show_instructions = not self.show_instructions
            self.blink_start_time = current_time

    def render(self, screen: pygame.Surface) -> None:
        """Render the landing screen."""
        # Fill the background
        screen.fill((30, 30, 30))

        # Use Background Image
        background = pygame.image.load(
            self.config.pygame_textures.backgrounds.dark
        )

        # Scale the background image to fit the screen
        screen.blit(
            pygame.transform.scale(background, screen.get_size()), (0, 0)
        )

        # Position title with a shadow for depth
        title_x: int = (
            screen.get_width() // 2 - self.title_text.get_width() // 2
        )
        title_y: int = (
            screen.get_height() // 2 - self.title_text.get_height() // 2
        )
        screen.blit(self.shadow_text, (title_x + 5, title_y + 5))
        screen.blit(self.title_text, (title_x, title_y))

        # Render instructions below the title if visible
        if self.show_instructions:
            instructions_x = (
                screen.get_width() // 2
                - self.instructions_text.get_width() // 2
            )
            instructions_y = title_y + self.title_text.get_height() + 20
            screen.blit(
                self.instructions_text, (instructions_x, instructions_y)
            )
