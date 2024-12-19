"""Board widget for Pygame UI."""

from typing import Tuple, List
import pygame

from src.ui.pygame.widgets.base_widget import Widget
from src.ui.utils.game_resource_loader import TextureLoader


class BoardWidget(Widget):
    def __init__(
        self,
        pos: Tuple[int, int],
        size: Tuple[int, int],
        z: int,
        textures: TextureLoader,
    ):
        super().__init__(pos, size, z)
        self.textures = textures

    def render(self, screen: pygame.Surface, **kwargs) -> None:
        """Render the board."""
        game_board: List[List[str]] = kwargs.get("game_board", None)
        if not game_board:
            return

        pygame.draw.rect(
            screen,
            (18, 90, 60),
            (
                self.position[0],
                self.position[1],
                len(game_board[0]) * self.size[0],
                len(game_board) * self.size[1],
            ),
        )

        for row_num, row in enumerate(game_board):
            for col_num, cell in enumerate(row):
                texture: pygame.Surface = pygame.image.load(
                    str(self.textures.textures[cell])
                )
                screen.blit(
                    texture,
                    (
                        self.position[0] + col_num * self.size[0],
                        self.position[1] + row_num * self.size[1],
                    ),
                )
