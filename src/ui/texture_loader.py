from pathlib import Path
from rich.console import Console
from rich.text import Text
from rich.style import Style
from src.config.settings import Config, get_config


class TextureLoader:
    KEYS: list[str] = [
        "wall",
        "snake_head",
        "snake_body",
        "green_apple",
        "red_apple",
        "empty",
    ]

    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.gui_mode: str = config.visual.modes.mode
        self.theme: str = config.visual.themes.default_theme
        self.textures = self.load_textures()

    def load_textures(self) -> dict[str, Text] | dict[str, Path]:
        return (
            self.load_ascii_textures()
            if self.gui_mode == "cli"
            else self.load_pygame_textures()
        )

    # <-- ASCII textures -->
    def load_ascii_textures(self) -> dict[str, Text]:
        styles: dict[str, Style] = (
            self.generate_ascii_style("white", "black")
            if self.theme == "dark"
            else self.generate_ascii_style("black", "grey89")
        )
        return {
            key: Text(getattr(self.config.ascii, key), style=styles[key])
            for key in self.KEYS
        }

    def generate_ascii_style(
        self, fg_color: str, bg_color: str
    ) -> dict[str, Style]:
        return {
            "wall": Style(color=fg_color, bgcolor=bg_color),
            "snake_head": Style(
                color="cyan" if fg_color == "white" else "bright_cyan",
                bgcolor=bg_color,
            ),
            "snake_body": Style(
                color="green" if fg_color == "white" else "cyan",
                bgcolor=bg_color,
            ),
            "green_apple": Style(
                color="yellow" if fg_color == "white" else "bright_green",
                bgcolor=bg_color,
            ),
            "red_apple": Style(color="red", bgcolor=bg_color),
            "empty": Style(color=fg_color, bgcolor=bg_color),
        }

    # <-- Pygame textures -->
    def load_pygame_textures(self) -> dict[str, Path]:
        return {
            key: getattr(
                getattr(self.config.pygame_textures, key), self.theme
            )
            for key in self.KEYS
        }

# <-- Testing -->
def print_textures(textures: dict):
    console = Console()
    for name, texture in textures.items():
        console.print(texture, f" {name}")


if __name__ == "__main__":
    config = get_config()
    textures = TextureLoader(config)

    print_textures(textures.textures)
