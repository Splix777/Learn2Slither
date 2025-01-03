"""Settings module for the Snake game configuration."""

import os
from typing import List, Literal
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_yaml import parse_yaml_raw_as


class ProjectConfig(BaseModel):
    """Project configuration model."""

    name: str
    description: str
    version: str
    author: str
    license: str
    repository: str
    environment: Literal["Development", "Production"]


class LoggingConfig(BaseModel):
    """Logging configuration model."""

    level: Literal["debug", "info", "warning", "error", "critical"]
    file: str


class BoardSize(BaseModel):
    """Board size configuration model."""

    width: int = Field(
        ge=10,
        le=100,
        description="Board width must be greater or equal to 10",
    )
    height: int = Field(
        ge=10,
        le=100,
        description="Board height must be greater or equal to 10",
    )


class MapConfig(BaseModel):
    """Map configuration model."""

    board_size: BoardSize
    green_apples: int = Field(
        gt=0, description="Starting green apples must be greater than 0"
    )
    red_apples: int = Field(
        gt=0, description="Starting red apples must be greater than 0"
    )


class SnakeModelPaths(BaseModel):
    """Snake model paths configuration model."""

    easy: Path
    medium: Path
    hard: Path
    expert: Path


class SnakeConfig(BaseModel):
    """Snake configuration model."""

    start_size: int = Field(
        gt=0,
        lt=10,
        description="Starting size must be greater than 0 and less than 10",
    )
    difficulty: SnakeModelPaths
    selected_difficulty: Path


class VisualModes(BaseModel):
    """Visual modes configuration model."""

    mode: Literal["cli", "pygame"]
    available_modes: List[Literal["cli", "pygame"]]


class ThemesConfig(BaseModel):
    """Themes configuration model."""

    selected_theme: Literal["dark", "light"]
    available_themes: List[Literal["light", "dark"]]


class VisualConfig(BaseModel):
    """Visual configuration model."""

    modes: VisualModes
    themes: ThemesConfig


class ASCIIConfig(BaseModel):
    """ASCII configuration model."""

    wall: str
    snake_head: str
    snake_body: str
    green_apple: str
    red_apple: str
    empty: str


class ThemedTextures(BaseModel):
    """Themed textures configuration model."""

    dark: Path
    light: Path


class PyGameTextures(BaseModel):
    """PyGame textures configuration model."""

    wall: ThemedTextures
    snake_head: ThemedTextures
    snake_body: ThemedTextures
    green_apple: ThemedTextures
    red_apple: ThemedTextures
    empty: ThemedTextures
    backgrounds: ThemedTextures
    texture_size: int


class PyGameAudio(BaseModel):
    home: Path
    game: Path


class Events(BaseModel):
    """Collisions configuration model."""

    death: int
    green_apple: int
    red_apple: int
    looping: int
    kill: int


class WinConditionsConfig(BaseModel):
    """Win conditions configuration model."""

    condition: Literal["max_snake_size", "last_snake_alive"]
    max_score: int = Field(
        gt=0, description="Max score must be greater than 0"
    )


class RulesConfig(BaseModel):
    """Rules configuration model."""

    events: Events
    steps_no_apple: int
    win_condition: WinConditionsConfig


class ExplorationConfig(BaseModel):
    """Exploration configuration model."""

    epsilon: float
    decay: float
    epsilon_min: float


class NeuralNetworkConfig(BaseModel):
    """Neural network configuration model."""

    batch_size: int
    epochs: int
    learning_rate: float
    exploration: ExplorationConfig
    gamma: float
    update_frequency: int
    input_shape: int
    output_shape: int
    memory_size: int
    patience: int
    min_delta: float


class PathsConfig(BaseModel):
    """Paths configuration model."""

    models: Path
    logs: Path
    outputs: Path
    textures: Path
    audio: Path


class Config(BaseModel):
    """Main configuration model."""

    project: ProjectConfig
    logging: LoggingConfig
    map: MapConfig
    snake: SnakeConfig
    visual: VisualConfig
    ascii: ASCIIConfig
    textures: PyGameTextures
    pygame_audio: PyGameAudio
    rules: RulesConfig
    nn: NeuralNetworkConfig
    paths: PathsConfig

    @model_validator(mode="after")
    def check_map_config(self: "Config") -> "Config":
        """Check if the map configuration is valid."""
        total_map_spaces: int = (
            self.map.board_size.width * self.map.board_size.height
        )

        walls: int = (
            (2 * self.map.board_size.width)
            + (2 * self.map.board_size.height)
            - 4
        )

        available_spaces: int = total_map_spaces - walls

        if available_spaces <= 0:
            raise ValueError("Not enough space for the map.")

        total_apples: int = self.map.green_apples + self.map.red_apples

        if total_apples > available_spaces:
            raise ValueError("Not enough space for the apples.")

        remaining_spaces: int = available_spaces - total_apples

        if self.snake.start_size > remaining_spaces:
            max_snake_size: int = remaining_spaces
            raise ValueError(f"Max snake size is {max_snake_size}")

        return self

    @model_validator(mode="after")
    def strings_to_paths(self: "Config") -> "Config":
        """Convert string paths to Path objects."""
        root_path: Path = Path(__file__).parent / "../../"

        self.paths.models = root_path / self.paths.models
        self.paths.logs = root_path / self.paths.logs
        self.paths.outputs = root_path / self.paths.outputs
        self.paths.textures = root_path / self.paths.textures
        self.paths.audio = root_path / self.paths.audio

        for texture in vars(self.textures).values():
            if isinstance(texture, ThemedTextures):
                texture.dark = self.paths.textures / texture.dark
                texture.light = self.paths.textures / texture.light

        for key, value in vars(self.pygame_audio).items():
            setattr(self.pygame_audio, key, self.paths.audio / value)

        for models in vars(self.snake).values():
            if isinstance(models, SnakeModelPaths):
                models.easy = self.paths.models / models.easy
                models.medium = self.paths.models / models.medium
                models.hard = self.paths.models / models.hard
                models.expert = self.paths.models / models.expert

        return self

    @model_validator(mode="after")
    def verify_textures_exist(self: "Config") -> "Config":
        """Verify that all texture files exist."""
        textures_to_check: List[tuple[str, list[str]]] = [
            ("wall", ["dark", "light"]),
            ("snake_head", ["dark", "light"]),
            ("snake_body", ["dark", "light"]),
            ("green_apple", ["dark", "light"]),
            ("red_apple", ["dark", "light"]),
            ("empty", ["dark", "light"]),
            ("backgrounds", ["dark", "light"]),
        ]

        for texture_name, sub_textures in textures_to_check:
            texture = getattr(self.textures, texture_name)
            for sub_texture in sub_textures:
                path = getattr(texture, sub_texture)
                if not path.exists():
                    raise FileNotFoundError(f"Texture file not found: {path}")

        return self

    @model_validator(mode="after")
    def verify_audio_exists(self: "Config") -> "Config":
        """Verify that all audio files exist."""
        audio_to_check: List[str] = [
            "home",
            "game",
        ]

        for audio_name in audio_to_check:
            path = getattr(self.pygame_audio, audio_name)
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")

        return self

    @model_validator(mode="after")
    def verify_ai_models_exist(self: "Config") -> "Config":
        """Verify that all AI models exist."""
        models_to_check: List[str] = [
            "easy",
            "medium",
            "hard",
            "expert",
        ]

        for model_name in models_to_check:
            path = getattr(self.snake.difficulty, model_name)
            if not path.exists():
                raise FileNotFoundError(f"AI model not found: {path}")

        self.snake.selected_difficulty = self.snake.difficulty.expert
        return self


def get_config() -> Config:
    """Load the configuration from the config.yaml file."""
    relative_path = "../../config.yaml"
    config_path: str = os.path.join(Path(__file__).parent, relative_path)

    try:
        with open(file=config_path, mode="r", encoding="utf-8") as f:
            raw_config: str = f.read()

        return parse_yaml_raw_as(model_type=Config, raw=raw_config)

    except ValidationError as e:
        print("Validation error occurred:", e)
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise ValidationError(f"Error loading configuration: {str(e)}") from e


# Singleton pattern
config: Config = get_config()
