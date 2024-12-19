from typing import List, Optional
from pathlib import Path

import pygame
from pygame.event import Event
from pygame.rect import Rect

from src.ui.pygame.screens.base_screen import BaseScreen
from src.config.settings import Config


class OptionsScreen(BaseScreen):
    def __init__(self, config: Config, theme: str, gui) -> None:
        self.config: Config = config
        self.theme: str = theme
        self.gui = gui
        self.font = pygame.font.Font(None, 50)

        music_option = "Music: On" if self.gui.music_on else "Music: Off"
        map_size_option = (
            f"Map Size: "
            f"{self.config.map.board_size.width}"
            f"x{self.config.map.board_size.height}"
        )
        self.difficulty_map = {
            Path(self.config.snake.difficulty.easy): "Easy",
            Path(self.config.snake.difficulty.medium): "Medium",
            Path(self.config.snake.difficulty.hard): "Hard",
            Path(self.config.snake.difficulty.expert): "Expert",
        }
        ai_difficulty_option = (
            f"AI Difficulty: "
            f"{self.difficulty_map[self.config.snake.selected_difficulty]}"
        )
        theme_option = (
            f"Theme: {self.config.visual.themes.selected_theme.capitalize()}"
        )

        self.options = [
            music_option,
            map_size_option,
            ai_difficulty_option,
            theme_option,
            "Back",
        ]
        self.selected_index = 0
        self.option_rects = []

        self.text_surfaces = []
        self.shadow_surfaces = []

        self.required_size = (1360, 960)
        self.current_resolution = self.required_size

        for option in self.options:
            text_surface = self.font.render(option, True, (255, 255, 255))
            shadow_surface = self.font.render(option, True, (50, 50, 50))
            self.text_surfaces.append(text_surface)
            self.shadow_surfaces.append(shadow_surface)

        self.next_screen = None

    def handle_input(self, events: List[Event]):
        """Handle input to navigate the options screen."""
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
                    self._select_option()
            if event.type == pygame.MOUSEMOTION:
                for i, rect in enumerate(self.option_rects):
                    if rect.collidepoint(event.pos):
                        self.selected_index = i
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for rect in self.option_rects:
                    if rect.collidepoint(event.pos):
                        self._select_option()

    def _select_option(self) -> None:
        """Handle the selection of an option."""
        if self.selected_index == 0:
            # Toggle Music
            self.gui.toggle_music()
            self.options[0] = (
                "Music: On" if self.gui.music_on else "Music: Off"
            )
        elif self.selected_index == 1:
            # Cycle Map Size
            sizes = [(10, 10), (20, 20), (30, 30)]
            current_size = (
                self.config.map.board_size.width,
                self.config.map.board_size.height,
            )
            next_size = sizes[(sizes.index(current_size) + 1) % len(sizes)]
            self.options[1] = f"Map Size: {next_size[0]}x{next_size[1]}"
            (
                self.config.map.board_size.width,
                self.config.map.board_size.height,
            ) = next_size
        elif self.selected_index == 2:
            # Cycle AI Difficulty
            difficulties = ["Easy", "Medium", "Hard", "Expert"]
            current_difficulty = self.difficulty_map[
                self.config.snake.selected_difficulty
            ]
            next_difficulty = difficulties[
                (difficulties.index(current_difficulty) + 1)
                % len(difficulties)
            ]
            self.options[2] = f"AI Difficulty: {next_difficulty}"
            self.config.snake.selected_difficulty = list(
                self.difficulty_map.keys()
            )[difficulties.index(next_difficulty)]
        elif self.selected_index == 3:
            # Toggle Theme
            themes = ["Light", "Dark"]
            current_theme = (
                self.config.visual.themes.selected_theme.capitalize()
            )
            next_theme = themes[
                (themes.index(current_theme) + 1) % len(themes)
            ]
            self.options[3] = f"Theme: {next_theme}"
            self.config.visual.themes.selected_theme = (
                "dark" if next_theme == "Dark" else "light"
            )
        elif self.selected_index == 4:
            self.next_screen = "home"

        self.text_surfaces = [
            self.font.render(option, True, (255, 255, 255))
            for option in self.options
        ]
        self.shadow_surfaces = [
            self.font.render(option, True, (50, 50, 50))
            for option in self.options
        ]

    def update(self) -> None:
        """Update the options screen."""
        pass

    def render(self, screen) -> None:
        """Render the options screen."""
        if screen.get_size() != self.current_resolution:
            self.current_resolution = screen.get_size()

        screen.fill((30, 30, 30))

        background: pygame.Surface = pygame.image.load(
            self.config.textures.backgrounds.dark
            if self.theme == "dark"
            else self.config.textures.backgrounds.light
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
        next_screen = self.next_screen
        self.next_screen = None
        return next_screen
