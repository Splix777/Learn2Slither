from enum import Enum
from typing import List


class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    @property
    def opposite(self) -> "Direction":
        """Returns the opposite direction."""
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }
        return opposites[self]

    @staticmethod
    def one_hot(direction: "Direction") -> List[int]:
        """
        Returns the one-hot encoding for a direction.
        
        Args:
            direction (Direction): The direction.

        Returns:
            List[int]: The one-hot encoding.
        """
        mapping: dict[Direction, list[int]] = {
            Direction.UP: [1, 0, 0, 0],
            Direction.DOWN: [0, 1, 0, 0],
            Direction.LEFT: [0, 0, 1, 0],
            Direction.RIGHT: [0, 0, 0, 1],
        }
        return mapping[direction]

    @staticmethod
    def from_int(direction: int) -> "Direction":
        """
        Returns the Direction from an integer.

        Args:
            direction (int): The integer representation of the direction.

        Returns:
            Direction: The direction.
        """
        if direction not in range(4):
            raise ValueError("Invalid direction.")
        mapping: List[Direction] = [
            Direction.UP,
            Direction.DOWN,
            Direction.LEFT,
            Direction.RIGHT,
        ]
        return mapping[direction]

    @staticmethod
    def from_one_hot(one_hot: List[int]) -> "Direction":
        """
        Returns the Direction from one-hot encoding.

        Args:
            one_hot (List[int]): The one-hot encoding.

        Returns:
            Direction: The direction.
        """
        if len(one_hot) != 4 or sum(one_hot) != 1:
            raise ValueError("Invalid one-hot encoding.")
        mapping: List[Direction] = [
            Direction.UP,
            Direction.DOWN,
            Direction.LEFT,
            Direction.RIGHT,
        ]
        index: int = one_hot.index(1)
        return mapping[index]

    @staticmethod
    def possible_directions(current: "Direction") -> List[int]:
        """Returns a binary list of valid directions excluding
        the opposite of the current.
        
        Args:
            current (Direction): The current direction.
        
        Returns:
            List[int]: A binary list of valid directions.
        """
        if not isinstance(current, Direction):
            raise ValueError("Invalid direction.")
        direction_indices: dict[Direction, int] = {
            Direction.UP: 0,
            Direction.DOWN: 1,
            Direction.LEFT: 2,
            Direction.RIGHT: 3,
        }
        possible: List[int] = [1] * 4
        possible[direction_indices[current.opposite]] = 0
        return possible


if __name__ == "__main__":
    # View what each propety returns
    print(Direction.UP.opposite)
    print(Direction.one_hot(Direction.UP))
    print(Direction.from_one_hot([1, 0, 0, 0]))
    print(Direction.possible_directions(Direction.UP))
