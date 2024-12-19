"""Widget base class abstract."""

import pygame
from typing import Tuple

from abc import ABC, abstractmethod


class Widget(ABC):
    def __init__(self, pos: Tuple[int, int], size: Tuple[int, int], z: int):
        self.position = pos
        self.size = size
        self.layer = z

    @abstractmethod
    def render(self, screen: pygame.Surface, **kwargs) -> None:
        """Render the widget on the screen."""
        pass
