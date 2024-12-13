"""Game manager module."""

from abc import ABC, abstractmethod
from typing import List


class GUI(ABC):
    """Abstract class for the game GUI."""

    @abstractmethod
    def run(self) -> None:
        """Run the GUI."""
        pass

    @abstractmethod
    def training_render(self, game_state: List[List[str]]) -> None:
        """Render the game environment during training."""
        pass