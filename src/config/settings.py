import os
from typing import List, Literal
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_yaml import parse_yaml_raw_as, to_yaml_str
from pathlib import Path


class ProjectConfig(BaseModel):
    name: str
    description: str
    version: str
    author: str
    license: str
    repository: str
    environment: Literal["Development", "Production"]


class LoggingConfig(BaseModel):
    level: Literal["debug", "info", "warning", "error", "critical"]
    file: str


class BoardSize(BaseModel):
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
    board_size: BoardSize
    green_apples: int = Field(
        gt=0, description="Starting green apples must be greater than 0"
    )
    red_apples: int = Field(
        gt=0, description="Starting red apples must be greater than 0"
    )
    snakes: int = Field(
        gt=0,
        le=4,
        description="Starting snakes must be greater than 0 and less than 4",
    )


class SnakeConfig(BaseModel):
    start_color: str
    start_speed: int
    start_size: int = Field(
        gt=0,
        lt=10,
        description="Starting size must be greater than 0 and less than 10",
    )
    ai_vision: int
    player_colors: List[str]
    ai_colors: List[str]
    max_speed: int


class VisualModes(BaseModel):
    mode: Literal["cli", "pygame"]
    available_modes: List[Literal["cli", "pygame"]]
    control: List[Literal["auto", "manual"]]
    multiplayer_modes: List[Literal["free_for_all", "team"]]


class ThemesConfig(BaseModel):
    default_theme: Literal["dark", "light"]
    available_themes: List[Literal["light", "dark"]]


class VisualConfig(BaseModel):
    modes: VisualModes
    speed: Literal["slow", "normal", "fast"]
    themes: ThemesConfig


class ASCIIConfig(BaseModel):
    wall: str
    snake_head: str
    snake_body: str
    green_apple: str
    red_apple: str
    empty: str


class ThemedTextures(BaseModel):
    dark: Path
    light: Path


class PyGameTextures(BaseModel):
    wall: ThemedTextures
    snake_head: ThemedTextures
    snake_body: ThemedTextures
    green_apple: ThemedTextures
    red_apple: ThemedTextures
    empty: ThemedTextures


class CollisionEffects(BaseModel):
    action: Literal["death", "grow", "shrink"]
    reward: int
    snake_effect: int


class Collisions(BaseModel):
    snake_collision: CollisionEffects
    wall_collision: CollisionEffects
    green_apple_collision: CollisionEffects
    red_apple_collision: CollisionEffects


class WinConditionsConfig(BaseModel):
    condition: Literal["max_snake_size", "last_snake_alive"]
    max_score: int = Field(
        gt=0, description="Max score must be greater than 0"
    )


class RulesConfig(BaseModel):
    collisions: Collisions
    win_condition: WinConditionsConfig


class LayerConfig(BaseModel):
    input_size: int
    hidden: List[int]
    output_size: int


class ActivationConfig(BaseModel):
    function: Literal["relu", "lrelu", "tanh", "sigmoid", "softmax"]
    output_function: Literal["sigmoid", "softmax"]


class ArchitectureConfig(BaseModel):
    layers: LayerConfig
    activation_function: ActivationConfig
    optimizer: Literal["adam", "sgd", "rmsprop"]
    loss_function: Literal["mse", "mae", "categorical_crossentropy"]


class ExplorationConfig(BaseModel):
    initial_rate: float
    decay: float
    minimum_rate: float


class TrainingConfig(BaseModel):
    batch_size: int
    epochs: int
    learning_rate: float
    exploration: ExplorationConfig
    discount_factor: float


class NeuralNetworkConfig(BaseModel):
    architecture: ArchitectureConfig
    training: TrainingConfig


class PathsConfig(BaseModel):
    models: Path
    logs: Path
    outputs: Path
    textures: Path


class Config(BaseModel):
    project: ProjectConfig
    logging: LoggingConfig
    map: MapConfig
    snake: SnakeConfig
    visual: VisualConfig
    ascii: ASCIIConfig
    pygame_textures: PyGameTextures
    rules: RulesConfig
    neural_network: NeuralNetworkConfig
    paths: PathsConfig

    @model_validator(mode="after")
    def check_map_config(self: "Config") -> "Config":
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
        root_path: Path = Path(__file__).parent / "../../"

        # Define paths
        self.paths.models = root_path / self.paths.models
        self.paths.logs = root_path / self.paths.logs
        self.paths.outputs = root_path / self.paths.outputs
        self.paths.textures = root_path / self.paths.textures

        # Iterate over the texture attributes and set the paths
        for texture in vars(self.pygame_textures).values():
            if isinstance(texture, ThemedTextures):
                texture.dark = self.paths.textures / texture.dark
                texture.light = self.paths.textures / texture.light

        return self

    @model_validator(mode="after")
    def verify_textures_exist(self: "Config") -> "Config":
        # Define the texture paths to check
        textures_to_check: List[tuple[str, list[str]]] = [
            ('wall', ['dark', 'light']),
            ('snake_head', ['dark', 'light']),
            ('snake_body', ['dark', 'light']),
            ('green_apple', ['dark', 'light']),
            ('red_apple', ['dark', 'light']),
            ('empty', ['dark', 'light']),
        ]

        # Check if all texture files exist
        for texture_name, sub_textures in textures_to_check:
            texture = getattr(self.pygame_textures, texture_name)
            for sub_texture in sub_textures:
                path = getattr(texture, sub_texture)
                if not path.exists():
                    raise FileNotFoundError(f"Texture file not found: {path}")
                
        return self

def get_config() -> Config:
    relative_path = "../../config.yaml"
    config_path: str = os.path.join(Path(__file__).parent, relative_path)

    try:
        # Load and parse YAML using pydantic-yaml
        with open(config_path, "r") as f:
            raw_config: str = f.read()

        # Validate the configuration
        config: Config = parse_yaml_raw_as(Config, raw_config)
        return config

    except ValidationError as e:
        print("Validation error occurred:", e)
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise ValidationError(f"Error loading configuration: {e}") from e


if __name__ == "__main__":
    try:
        config: Config = get_config()
        # print(config)
        print(to_yaml_str(config))
    except Exception as e:
        print(e)
