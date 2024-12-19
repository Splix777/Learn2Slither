"""Game manager module."""

from abc import ABC, abstractmethod


class GUI(ABC):
    """Abstract class for the game GUI."""

    @abstractmethod
    def run(self) -> None:
        """Run the GUI."""
        pass
