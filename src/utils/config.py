import os
from typing import List, Literal, Dict
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_yaml import parse_yaml_raw_as
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
        ge=10, description="Board width must be greater or equal to 10"
    )
    height: int = Field(
        ge=10, description="Board height must be greater or equal to 10"
    )


class MapConfig(BaseModel):
    board_size: BoardSize
    starting_green_apples: int = Field(
        gt=0, description="Starting green apples must be greater than 0"
    )
    starting_red_apples: int = Field(
        gt=0, description="Starting red apples must be greater than 0"
    )


class SnakeConfig(BaseModel):
    start_color: str
    start_speed: int
    start_size: int = Field(
        gt=0,
        lt=10,
        description="Starting size must be greater than 0 and less than 10",
    )
    start_vision: int


class VisualModes(BaseModel):
    type: List[Literal["cli", "pygame"]]
    mode: Literal["cli", "pygame"]
    control: List[Literal["auto", "manual"]]


class VisualConfig(BaseModel):
    modes: VisualModes
    speed: Literal["slow", "normal", "fast"]


class CollisionEffectsConfig(BaseModel):
    action: Literal["death", "grow", "shrink"]
    reward: int
    snake_effect: int


class CollisionsConfig(BaseModel):
    snake_collision: CollisionEffectsConfig
    wall_collision: CollisionEffectsConfig
    green_apple_collision: CollisionEffectsConfig
    red_apple_collision: CollisionEffectsConfig


class WinConditionsConfig(BaseModel):
    condition: Literal["max_score", "max_snake_size"]
    max_score: int = Field(
        gt=0, description="Max score must be greater than 0"
    )


class RulesConfig(BaseModel):
    collisions: CollisionsConfig
    win_condition: WinConditionsConfig


class LearningParameters(BaseModel):
    learning_rate: float = Field(ge=0.0, le=1.0)
    discount_factor: float = Field(ge=0.0, le=1.0)
    exploration_rate: float = Field(ge=0.0, le=1.0)
    exploration_decay: float = Field(ge=0.0, le=1.0)
    min_exploration_rate: float = Field(ge=0.0, le=1.0)


class PathsConfig(BaseModel):
    models: str
    logs: str
    outputs: str


class ASCIIConfig(BaseModel):
    wall: str
    snake_head: str
    snake_body: str
    green_apple: str
    red_apple: str
    empty: str


class Config(BaseModel):
    project: ProjectConfig
    logging: LoggingConfig
    map: MapConfig
    snake: SnakeConfig
    visual: VisualConfig
    ASCII: ASCIIConfig
    rules: RulesConfig
    allowed_models: Dict[str, List[str]]
    learning_parameters: LearningParameters
    paths: PathsConfig

    @model_validator(mode="after")
    def check_map_config(self: "Config") -> "Config":
        total_map_spaces: int = (
            self.map.board_size.width
            * self.map.board_size.height)

        walls: int = (
            (2 * self.map.board_size.width)
            + (2 * self.map.board_size.height)
            - 4
        )

        available_spaces: int = total_map_spaces - walls

        if available_spaces <= 0:
            raise ValueError("Not enough space for the map.")

        total_apples: int = (
            self.map.starting_green_apples
            + self.map.starting_red_apples
        )

        if total_apples > available_spaces:
            raise ValueError("Not enough space for the apples.")

        remaining_spaces: int = available_spaces - total_apples

        if self.snake.start_size > remaining_spaces:
            max_snake_size: int = remaining_spaces
            raise ValueError(f"Max snake size is {max_snake_size}")

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
        raise e
    except Exception as e:
        raise ValidationError(f"Error loading configuration: {e}") from e


if __name__ == "__main__":
    try:
        config: Config = get_config()
        print(config)
        # print(to_yaml_str(config))
    except Exception as e:
        print(e)
