from abc import ABC, abstractmethod

class BaseScreen(ABC):
    @abstractmethod
    def handle_input(self, events):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def render(self, screen):
        pass

    @abstractmethod
    def get_next_screen(self):
        return None
