import unittest

from src.game.models.directions import Direction


class TestDirection(unittest.TestCase):
    def test_one_hot(self) -> None:
        self.assertEqual(Direction.one_hot(Direction.UP), [1, 0, 0, 0])
        self.assertEqual(Direction.one_hot(Direction.DOWN), [0, 1, 0, 0])
        self.assertEqual(Direction.one_hot(Direction.LEFT), [0, 0, 1, 0])
        self.assertEqual(Direction.one_hot(Direction.RIGHT), [0, 0, 0, 1])

    def test_from_one_hot(self) -> None:
        self.assertEqual(Direction.from_one_hot([1, 0, 0, 0]), Direction.UP)
        self.assertEqual(Direction.from_one_hot([0, 1, 0, 0]), Direction.DOWN)
        self.assertEqual(Direction.from_one_hot([0, 0, 1, 0]), Direction.LEFT)
        self.assertEqual(
            Direction.from_one_hot([0, 0, 0, 1]), Direction.RIGHT
        )

    def test_possible_directions(self) -> None:
        self.assertEqual(
            Direction.possible_directions(Direction.UP), [1, 0, 1, 1]
        )
        self.assertEqual(
            Direction.possible_directions(Direction.DOWN), [0, 1, 1, 1]
        )
        self.assertEqual(
            Direction.possible_directions(Direction.LEFT), [1, 1, 1, 0]
        )
        self.assertEqual(
            Direction.possible_directions(Direction.RIGHT), [1, 1, 0, 1]
        )

    def test_opposite(self) -> None:
        self.assertEqual(Direction.UP.opposite, Direction.DOWN)
        self.assertEqual(Direction.DOWN.opposite, Direction.UP)
        self.assertEqual(Direction.LEFT.opposite, Direction.RIGHT)
        self.assertEqual(Direction.RIGHT.opposite, Direction.LEFT)

    def test_invalid_one_hot(self) -> None:
        with self.assertRaises(ValueError):
            Direction.from_one_hot([1, 0, 0])
        with self.assertRaises(ValueError):
            Direction.from_one_hot([1, 1, 0, 0])
        with self.assertRaises(ValueError):
            Direction.from_one_hot([0, 0, 0, 0])

    def test_invalid_direction(self) -> None:
        with self.assertRaises(ValueError):
            Direction.possible_directions("invalid")


if __name__ == "__main__":
    unittest.main()
