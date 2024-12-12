from typing import List

import pygame
from pygame.event import Event

from src.ui.pygame.screens.screen import BaseScreen
from src.config.settings import Config


class HomeScreen(BaseScreen):
	def __init__(self, config: Config) -> None:
		self.font = pygame.font.Font(None, 36)
		self.title_text: pygame.Surface = self.font.render("Press Enter to Start", True, (255, 255, 255))

	def handle_input(self, events: List[Event]) -> str:
		for event in events:
			if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
				return "game"
		return ""

	def update(self) -> None:
		pass

	def render(self, screen) -> None:
		screen.fill((0, 0, 0))  # Black background
		screen.blit(self.title_text, (100, 100))  # Render title