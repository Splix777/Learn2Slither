from typing import List, Optional

import pygame
from pygame.event import Event
from pygame.rect import Rect

from src.ui.pygame.screens.base_screen import BaseScreen
from src.config.settings import Config


class HomeScreen(BaseScreen):
    def __init__(self, config: Config, theme: str) -> None:
        self.config: Config = config
        self.theme: str = theme
        self.font = pygame.font.Font(None, 50)
        self.options = ["Human vs AI", "AI Solo", "Options"]
        self.selected_index = 0
        self.option_rects = []

        self.text_surfaces = []
        self.shadow_surfaces = []

        self.required_size = (800, 600)
        self.current_resolution = self.required_size

        for option in self.options:
            text_surface = self.font.render(option, True, (255, 255, 255))
            shadow_surface = self.font.render(option, True, (50, 50, 50))
            self.text_surfaces.append(text_surface)
            self.shadow_surfaces.append(shadow_surface)

        self.next_screen = None

    def handle_input(self, events: List[Event]):
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
                    self.next_screen = (
                        self.options[self.selected_index]
                        .lower()
                        .replace(" ", "_")
                    )

            if event.type == pygame.MOUSEMOTION:
                for i, rect in enumerate(self.option_rects):
                    if rect.collidepoint(event.pos):
                        self.selected_index = i

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, rect in enumerate(self.option_rects):
                    if rect.collidepoint(event.pos):
                        self.next_screen = (
                            self.options[i].lower().replace(" ", "_")
                        )

    def update(self) -> None:
        """Update the home screen."""
        pass

    def render(self, screen) -> None:
        """Render the home screen."""
        if screen.get_size() != self.current_resolution:
            self.current_resolution = screen.get_size()

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

    def get_next_screen(self) -> Optional[str]:
        return self.next_screen
