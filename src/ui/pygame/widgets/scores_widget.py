"""ScoresWidget class definition."""

from typing import Tuple
import pygame

from src.ui.pygame.widgets.base_widget import Widget
from src.ui.utils.game_resource_loader import TextureLoader


class ScoreWidget(Widget):
    def __init__(
        self,
        pos: Tuple[int, int],
        size: Tuple[int, int],
        z: int,
        textures: TextureLoader,
        snake_id: int,
    ):
        """Initialize the ScoreWidget."""
        super().__init__(pos, size, z)
        self.textures = textures
        self.snake_id = snake_id
        self.apple_image = pygame.image.load(
            str(self.textures.textures["green_apple"])
        )

    def render(self, screen: pygame.Surface, **kwargs) -> None:
        """Render the score widget on the screen."""
        score: int = kwargs.get("score", 0)

        # Draw rounded rectangle outline
        pygame.draw.rect(
            screen,
            (255, 255, 255),  # White outline color
            (*self.position, *self.size),
            width=2,  # Thickness of the outline
            border_radius=10,  # Rounding radius
        )

        # Blit the apple image inside the box
        apple_rect = self.apple_image.get_rect()
        apple_rect.topleft = (
            self.position[0] + 10,
            self.position[1] + 10,
        )
        screen.blit(self.apple_image, apple_rect)

        # Render the text: Snake ID and Score
        font = pygame.font.Font(None, 36)
        snake_text = font.render(
            f"Snake {self.snake_id}", True, (255, 255, 255)
        )
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))

        # Position the text
        text_padding = 10
        screen.blit(
            snake_text,
            (
                self.position[0] + apple_rect.width + 20,
                self.position[1] + text_padding,
            ),
        )
        screen.blit(
            score_text,
            (
                self.position[0] + apple_rect.width + 20,
                self.position[1] + text_padding + 40,
            ),
        )
