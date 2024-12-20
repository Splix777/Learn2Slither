"""Snake class module."""

import random
from typing import Tuple, List, Optional
from contextlib import suppress

import torch

from src.game.models.directions import Direction
from src.ai.agent import Agent
from src.config.settings import Config


class Snake:
    def __init__(
        self,
        id: int,
        config: Config,
        brain: Optional[Agent] = None,
    ) -> None:
        self.id: int = id
        self.config: Config = config
        self.size: int = config.snake.start_size
        self.brain: Optional[Agent] = brain
        self.alive: bool = True
        self.kills: int = 0
        self.reward: int = 0
        self.red_apples_eaten: int = 0
        self.green_apples_eaten: int = 0
        self.steps_without_food: int = 0
        self.head: Tuple[int, int] = (0, 0)
        self.body: List[Tuple[int, int]] = []
        self.movement_direction: Direction = Direction.UP
        self.initialized: bool = False

    def __str__(self) -> str:
        """Return the snake's string representation."""
        return f"Snake {self.id} - Size: {self.size} - Alive: {self.alive}"

    def __repr__(self) -> str:
        """Return the snake's representation."""
        return f"Snake {self.id} - Size: {self.size} - Alive: {self.alive}"

    # Initialization
    def initialize(
        self, start_pos: Tuple[int, int], start_dir: Direction
    ) -> None:
        """Reset the snake to its initial state."""
        self.initialized = False
        self.size = self.config.snake.start_size
        self.alive = True
        self.kills = 0
        self.reward = 0
        self.red_apples_eaten = 0
        self.green_apples_eaten = 0
        self.steps_without_food = 0
        self.head = start_pos
        self.body = [self.head]
        self.movement_direction = start_dir
        self.create_snake_body()
        self.initialized = True

    def create_snake_body(self) -> None:
        """Initialize the snake body."""
        for body in range(1, self.size):
            segment_offset: Tuple[int, int] = (
                body * self.movement_direction.value[0],
                body * self.movement_direction.value[1],
            )
            snake_segment: Tuple[int, int] = (
                self.head[0] - segment_offset[0],
                self.head[1] - segment_offset[1],
            )
            self.body.append(snake_segment)
        self.initialized = True

    # Movement methods
    def snake_controller(self, direction: List[int] | int) -> None:
        """Change the snake's movement direction."""
        new_direction: Direction = self.movement_direction

        if isinstance(direction, list):
            with suppress(ValueError):
                new_direction = Direction.from_one_hot(direction)
        elif isinstance(direction, int):
            with suppress(ValueError):
                new_direction = Direction.from_int(direction)

        if new_direction != self.movement_direction.opposite:
            self.movement_direction = new_direction

    def move(
        self, state: Optional[torch.Tensor] = None, learn: bool = False
    ) -> List[int]:
        """Move the snake in the current direction."""
        if not self.alive:
            return Direction.one_hot(self.movement_direction)

        self.steps_without_food += 1
        if self.brain and state is not None:
            self.snake_controller(self.brain.act(state, learn))
            if self.starving():
                self.snake_controller(random.choice([0, 1, 2, 3]))

        new_head: Tuple[int, int] = (
            self.head[0] + self.movement_direction.value[0],
            self.head[1] + self.movement_direction.value[1],
        )

        self.body.insert(0, new_head)
        self.head = new_head

        while len(self.body) > self.size:
            self.body.pop()

        return Direction.one_hot(self.movement_direction)

    # Event methods
    def eat_green_apple(self) -> None:
        """Increase the snake's size and reward."""
        self.size += 1
        self.green_apples_eaten += 1
        self.reward = self.config.rules.events.green_apple
        self.steps_without_food = 0

    def eat_red_apple(self) -> None:
        """Decrease the snake's size and reward."""
        self.size -= 1
        if self.size == 0:
            self.death()
        self.red_apples_eaten += 1
        self.reward = self.config.rules.events.red_apple
        self.steps_without_food = 0

    def looping(self) -> None:
        """Apply the looping penalty to the snake."""
        self.reward = self.config.rules.events.looping

    def kill(self) -> None:
        """Decrease the snake's size and reward."""
        self.kills += 1
        self.reward = self.config.rules.events.kill

    def death(self) -> None:
        """Delete the snake."""
        self.head = (0, 0)
        self.body = []
        self.size = 0
        self.alive = False
        self.reward = self.config.rules.events.death

    def starving(self) -> bool:
        """Apply the starving penalty to the snake."""
        if self.steps_without_food > self.config.rules.steps_no_apple:
            self.steps_without_food = 0
            return True
        return False

    def reset_rewards(self) -> None:
        """Reset the snake's reward."""
        self.reward = 0

    # Properties
    @property
    def one_hot_direction(self) -> List[int]:
        """One-hot representation of snake's movement."""
        return Direction.one_hot(self.movement_direction)

    @property
    def one_hot_options(self) -> List[int]:
        """One-hot representation of snake's possible movement."""
        return Direction.possible_directions(self.movement_direction)
