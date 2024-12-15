from typing import Tuple, List, Optional
from contextlib import suppress

import torch

from src.game.models.directions import Direction
from src.ai.agent import Brain


class Snake:
    def __init__(
        self,
        id: int,
        size: int = 3,
        brain: Optional[Brain] = None,
    ) -> None:
        self.id: int = id
        self.size: int = size
        self.brain: Optional[Brain] = brain
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

    def __repr__(self) -> str:
        """Return the snake's representation."""
        return f"Snake {self.id} - Size: {self.size} - Alive: {self.alive}"

    def init(self, start_pos: Tuple[int, int], start_dir: Direction):
        """Initialize the snake body."""
        self.head = start_pos
        self.body.append(self.head)
        self.movement_direction = start_dir
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

    def reset(self, start_pos: Tuple[int, int], start_dir: Direction) -> None:
        """Reset the snake to its initial state."""
        self.size = 3
        self.alive = True
        self.kills = 0
        self.reward = 0
        self.red_apples_eaten = 0
        self.green_apples_eaten = 0
        self.steps_without_food = 0
        self.head = start_pos
        self.body = [self.head]
        self.movement_direction = start_dir
        self.initialized = False
        self.init(start_pos, start_dir)

    # <-- Movement methods -->
    def snake_controller(self, direction: List[int] | int) -> None:
        """
        Change the snake's movement direction. Suppresses ValueError
        if the new direction is the opposite of the current direction
        or if the new direction is invalid.

        Args:
            direction (List[int] | int): The new direction of the snake.
        """
        new_direction: Direction = self.movement_direction

        if isinstance(direction, list):
            with suppress(ValueError):
                new_direction = Direction.from_one_hot(direction)
        elif isinstance(direction, int):
            with suppress(ValueError):
                new_direction = Direction.from_int(direction)

        if new_direction != self.movement_direction.opposite:
            self.movement_direction = new_direction

    def move(self, state: Optional[torch.Tensor] = None, learn: bool = False) -> List[int]:
        """Move the snake in the current direction."""
        if not self.alive:
            return Direction.one_hot(self.movement_direction)
        
        if self.brain and state is not None:
            self.snake_controller(
                self.brain.act(state) 
                if learn 
                else self.brain.choose_action(state)
            )

        new_head: Tuple[int, int] = (
            self.head[0] + self.movement_direction.value[0],
            self.head[1] + self.movement_direction.value[1],
        )

        self.body.insert(0, new_head)
        self.head = new_head

        while len(self.body) > self.size:
            self.body.pop()

        return Direction.one_hot(self.movement_direction)

    # <-- Size methods -->
    def grow(self) -> None:
        """Increase the snake's size by one."""
        self.size += 1

    def shrink(self) -> None:
        """Decrease the snake's size by one."""
        self.size -= 1
        if self.size == 0:
            self.alive = False

    # <-- Reward methods -->
    def reset_reward(self) -> None:
        """Reset the snake's reward."""
        self.reward = 0

    def update_reward(self, event: str, reward: int) -> None:
        """
        Update the snake's reward based on the event.

        Args:
            event (str): The event that occurred.
            reward (int): The reward to add.
        """
        if event in {"death", "green_apple", "red_apple", "looping"}:
            # print(f"Snake {self.id} - Event: {event} - Reward: {reward}")
            self.reward = reward

    def delete(self) -> None:
        """Delete the snake."""
        self.body = []
        self.size = 0
        self.alive = False

    # <-- Properties -->
    @property
    def direction_one_hot(self) -> List[int]:
        """One-hot representation of snake's movement."""
        return Direction.one_hot(self.movement_direction)

    @property
    def possible_directions_one_hot(self) -> List[int]:
        """One-hot representation of snake's possible movement."""
        return Direction.possible_directions(self.movement_direction)
