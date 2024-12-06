import os
from enum import Enum
from typing import Dict, List
import asyncio
from src.utils.config import Config, get_config
from src.game_objects.map import Map


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class Snake:
    map: Map
    snakes: List["Snake"] = []
    _id_counter: int = 0

    def __init__(self, config: Config, **kwargs):
        self.id: int = Snake._id_counter
        Snake._id_counter += 1
        self.config = config
        self.name: str = kwargs.get("name", f"Snake_{self.id}")
        self.kills: int = 0
        self.alive: bool = True
        self.head: Dict[str, int] = {}
        self.body: List[Dict[str, int]] = []
        self.size: int = config.snake.start_size
        self.color: str = config.snake.start_color
        self.speed: float = config.snake.start_speed
        self.current_direction: Direction = Direction.DOWN
        self.directions_map: Dict[str, Direction] = {
            "W": Direction.UP,
            "S": Direction.DOWN,
            "A": Direction.LEFT,
            "D": Direction.RIGHT,
        }
        self.initialize()
        Snake.snakes.append(self)

    def initialize(self) -> None:
        """Initialize the snake's head and body on the map."""
        self.head = Snake.map.get_random_empty_position(Snake.map.map)
        self.body = [self.head]
        Snake.map.map[self.head["y"]][self.head["x"]] = self.config.ASCII.snake_head
        self._add_body_parts(self.head, self.size - 1)

    def _add_body_parts(self, curr_part: Dict[str, int], remaining_size: int) -> None:
        """Recursively add body parts to the snake."""
        if remaining_size <= 0:
            return

        for direction in Direction:
            x, y = curr_part["x"] + direction.value[0], curr_part["y"] + direction.value[1]
            if Snake.map.is_empty(x, y):
                new_part = {"x": x, "y": y}
                self.body.append(new_part)
                Snake.map.map[new_part["y"]][new_part["x"]] = self.config.ASCII.snake_body
                self._add_body_parts(new_part, remaining_size - 1)
                return

    async def move(self) -> None:
        """Move the snake in its current direction."""
        while self.alive:
            await asyncio.sleep(self.speed)
            new_x = self.head["x"] + self.current_direction.value[0]
            new_y = self.head["y"] + self.current_direction.value[1]

            if self._check_collision(new_x, new_y):
                break
            if Snake.map.is_apple(new_x, new_y):
                self._handle_apple(new_x, new_y)
            else:
                self._update_body_and_tail(new_x, new_y, False, False)

    def _update_body_and_tail(self, x: int, y: int, grow: bool, shrink: bool) -> None:
        """Update the body and tail when the snake moves."""
        self.head = {"x": x, "y": y}
        self.body.insert(0, self.head)
        Snake.map.map[y][x] = self.config.ASCII.snake_head

        if len(self.body) > 1:
            old_head = self.body[1]
            Snake.map.map[old_head["y"]][old_head["x"]] = self.config.ASCII.snake_body

        if not grow:
            self._remove_tail()
        if shrink:
            self._remove_tail()
            self.size -= 1
            if self.size <= 0:
                self.alive = False

    def _remove_tail(self) -> None:
        """Remove the snake's tail from the map."""
        tail = self.body.pop()
        Snake.map.map[tail["y"]][tail["x"]] = self.config.ASCII.empty

    def _handle_apple(self, x: int, y: int) -> None:
        """Handle logic when the snake eats an apple."""
        if Snake.map.is_green_apple(x, y):
            Snake.map.current_green_apples -= 1
            self._update_body_and_tail(x, y, True, False)
        elif Snake.map.is_red_apple(x, y):
            Snake.map.current_red_apples -= 1
            self._update_body_and_tail(x, y, False, True)
        Snake.map.add_apples(Snake.map.map)

    def _check_collision(self, x: int, y: int) -> bool:
        """Check if the snake has collided with a wall or itself."""
        if (
            Snake.map.is_wall(x, y)
            or self.is_own_body(x, y)
            or self.is_other_snake(x, y)
        ):
            self.alive = False
            Snake.snakes.remove(self)
            for part in self.body:
                Snake.map.map[part["y"]][part["x"]] = self.config.ASCII.empty

            if self.is_other_snake(x, y):
                for snake in Snake.snakes:
                    if snake.is_other_snake(x, y):
                        snake.kills += 1
                
            return True

        return False

    def is_own_body(self, x: int, y: int) -> bool:
        return {"x": x, "y": y} in self.body

    def is_other_snake(self, x: int, y: int) -> bool:
        return any({"x": x, "y": y} in snake.body and snake.id != self.id for snake in Snake.snakes)


async def main():
    """Main asynchronous game loop."""
    config: Config = get_config()
    Snake.map = Map(config)

    # Initialize snakes
    snakes = [Snake(config, name=f"Player_{i+1}") for i in range(2)]
    Snake.map.add_apples(Snake.map.map)

    # Start all snake movements concurrently
    tasks = [asyncio.create_task(snake.move()) for snake in snakes]

    while any(snake.alive for snake in snakes):
        os.system("cls" if os.name == "nt" else "clear")
        Snake.map.display_map()
        await asyncio.sleep(0.1)

    # Cancel tasks if all snakes are dead
    for task in tasks:
        task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
