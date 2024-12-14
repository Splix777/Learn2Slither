from typing import List, Optional

import pygame
from pygame.event import Event
from pygame.rect import Rect

from src.ui.pygame.screens.screen import BaseScreen
from src.config.settings import Config


class HomeScreen(BaseScreen):
    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.font = pygame.font.Font(None, 50)  # Larger font for buttons
        self.options = ["Human vs AI", "AI Solo"]
        self.selected_index = 0
        self.option_rects = []

        # Store rendered text surfaces and their shadows
        self.text_surfaces = []
        self.shadow_surfaces = []

        # Requested screen size
        self.requested_screen_size = None
        self.request_screen_resize((640, 480))

        for option in self.options:
            text_surface = self.font.render(option, True, (255, 255, 255))
            shadow_surface = self.font.render(option, True, (50, 50, 50))
            self.text_surfaces.append(text_surface)
            self.shadow_surfaces.append(shadow_surface)

    def handle_input(self, events: List[Event]) -> str:
        """Handle input to navigate to the home screen."""
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    self.selected_index = (self.selected_index + 1) % len(
                        self.options
                    )
                elif event.key == pygame.K_UP:
                    self.selected_index = (self.selected_index - 1) % len(
                        self.options
                    )
                elif event.key == pygame.K_RETURN:
                    return (
                        self.options[self.selected_index]
                        .lower()
                        .replace(" ", "_")
                    )

            if event.type == pygame.MOUSEMOTION:
                for i, rect in enumerate(self.option_rects):
                    if rect.collidepoint(event.pos):
                        self.selected_index = i

            if (
                event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
            ):  # Left click
                for i, rect in enumerate(self.option_rects):
                    if rect.collidepoint(event.pos):
                        return self.options[i].lower().replace(" ", "_")

        return ""

    def update(self) -> None:
        """Update the home screen."""
        pass

    def render(self, screen) -> None:
        """Render the home screen."""
        if screen.get_size() != (640, 480):
            self.request_screen_resize((640, 480))

        screen.fill((30, 30, 30))

        background: pygame.Surface = pygame.image.load(
            self.config.pygame_textures.backgrounds.dark
        )
        screen.blit(
            pygame.transform.scale(background, screen.get_size()), (0, 0)
        )

        screen_width, screen_height = screen.get_size()
        center_x = screen_width // 2
        start_y = screen_height // 2 - (len(self.options) * 50 // 2)

        self.option_rects = []

        for i, (text_surface, shadow_surface) in enumerate(
            zip(self.text_surfaces, self.shadow_surfaces)
        ):
            text_width, text_height = text_surface.get_size()
            x = center_x - text_width // 2
            y = start_y + i * 70

            shadow_x = x + 2
            shadow_y = y + 2

            if i == self.selected_index:
                pygame.draw.rect(
                    screen,
                    (100, 100, 100),
                    Rect(x - 10, y - 10, text_width + 20, text_height + 20),
                    border_radius=10,
                )

            screen.blit(shadow_surface, (shadow_x, shadow_y))
            screen.blit(text_surface, (x, y))

            self.option_rects.append(Rect(x, y, text_width, text_height))

    def request_screen_resize(self, new_size: tuple[int, int]) -> None:
        """Request a screen size update."""
        self.requested_screen_size = new_size

    def get_requested_screen_size(self) -> Optional[tuple[int, int]]:
        """Return the requested screen size and reset the request."""
        size = self.requested_screen_size
        self.requested_screen_size = None
        return size
