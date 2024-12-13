from typing import Tuple, List
from contextlib import suppress

from src.game.models.directions import Direction


class Snake:
    def __init__(
        self,
        id: int,
        size: int,
        start_pos: Tuple[int, int],
        start_dir: Direction,
    ) -> None:
        self.id: int = id
        self.size: int = size
        self.alive: bool = True
        self.kills: int = 0
        self.reward: int = 0
        self.red_apples_eaten: int = 0
        self.green_apples_eaten: int = 0
        self.head: Tuple[int, int] = start_pos
        self.body: List[Tuple[int, int]] = [self.head]
        self.movement_direction: Direction = start_dir
        self.initialize()

    def __repr__(self) -> str:
        """Return the snake's representation."""
        return f"Snake {self.id} - Size: {self.size} - Alive: {self.alive}"

    def initialize(self) -> None:
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

    def move(self) -> List[Tuple[int, int]]:
        """Move the snake in the current direction."""
        removed_tails: List[Tuple[int, int]] = []
        if not self.alive:
            return removed_tails

        new_head: Tuple[int, int] = (
            self.head[0] + self.movement_direction.value[0],
            self.head[1] + self.movement_direction.value[1],
        )

        self.body.insert(0, new_head)
        self.head = new_head

        while len(self.body) > self.size:
            removed_tails.append(self.body.pop())

        return removed_tails

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
        if event in {"death", "green_apple", "red_apple"}:
            self.reward = reward

    # <-- Properties -->
    @property
    def direction_one_hot(self) -> List[int]:
        """One-hot representation of snake's movement."""
        return Direction.one_hot(self.movement_direction)

    @property
    def possible_directions_one_hot(self) -> List[int]:
        """One-hot representation of snake's possible movement."""
        return Direction.possible_directions(self.movement_direction)
