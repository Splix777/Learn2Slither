from abc import ABC, abstractmethod


class BaseScreen(ABC):
    @abstractmethod
    def handle_input(self, events) -> str:
        """Process input and return the next screen name, if needed."""
        pass

    @abstractmethod
    def update(self) -> None:
        """Update the state of the screen."""
        pass

    @abstractmethod
    def render(self, screen) -> None:
        """Render the screen's visuals."""
        pass
