from typing import Optional, List

import pygame
from pygame.event import Event

from src.ui.pygame.screens.screen import BaseScreen
from src.config.settings import Config


class GameScreen(BaseScreen):
    def __init__(self, config: Config) -> None:
        self.game_state: Optional[List[List[str]]] = None

    def handle_input(self, events: List[Event]) -> str:
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "home"
        return ""

    def update(self) -> None:
        # Update game logic here
        # For now, this is just a placeholder
        pass

    def render(self, screen) -> None:
        screen.fill((0, 0, 0))  # Clear the screen

    def update_map_state(self, game_state) -> None:
        self.game_state = game_state
