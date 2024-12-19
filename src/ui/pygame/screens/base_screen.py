"""BaseScreen class abstract definition."""

from typing import Optional

from abc import ABC, abstractmethod


class BaseScreen(ABC):
    @abstractmethod
    def handle_input(self, events) -> None:
        pass

    @abstractmethod
    def update(self) -> None:
        pass

    @abstractmethod
    def render(self, screen) -> None:
        pass

    @abstractmethod
    def get_next_screen(self) -> Optional[str]:
        return None
