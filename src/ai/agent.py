import time
import random
from typing import List, Optional
from collections import deque

import torch

from src.game.snake import Snake
from src.utils.plotter import Plotter
from src.ai.dqn_snake import DQNSnake
from src.game.interpreter import Interpreter
from src.config.settings import Config, config
from src.ui.texture_loader import TextureLoader


class Agent:
    def __init__(
        self,
        config: Config,
        interpreter: Interpreter,
        plotter: Plotter,
        model_path: Optional[str] = None,
    ) -> None:
        """
        Initializes the AI agent for Snake.
        Args:
            config (Config): Configuration settings.
            interpreter (Interpreter): Game interpreter.
            plotter (Plotter): Plotter for visualizing training.
            model_path (str, optional): Path to load pre-trained model.
        """
        self.config: Config = config
        self.interpreter: Interpreter = interpreter
        self.plotter: Plotter = plotter
        self.gamma: float = config.nn.training.gamma
        self.batch_size: int = config.nn.training.batch_size
        self.epochs: int = config.nn.training.epochs
        self.decay: float = config.nn.training.exploration.decay
        self.epsilon_min: float = config.nn.training.exploration.epsilon_min

        # Initialize separate models and replay buffers for each snake
        self.snakes_data = [
            {
                "model": DQNSnake(interpreter.get_state_size(), 4, config),
                "goal_model": DQNSnake(interpreter.get_state_size(), 4, config),
                "memory": deque(maxlen=100_000),
                "epsilon": config.nn.training.exploration.epsilon,
            }
            for _ in range(len(interpreter.board.snakes))
        ]

        # Load pre-trained model if provided
        if model_path:
            for snake_data in self.snakes_data:
                snake_data["model"].load(model_path)

        self.update_target_models()

    def update_target_models(self) -> None:
        """
        Updates the target models for all snakes.
        """
        for snake_data in self.snakes_data:
            snake_data["goal_model"].load_state_dict(snake_data["model"].state_dict())

    def remember(self, snake_data, state, action, reward, next_state, done) -> None:
        """
        Stores an experience in the replay buffer for a specific snake.
        """
        snake_data["memory"].append((state, action, reward, next_state, done))

    def act(self, state, snake_data) -> list[int]:
        """
        Chooses an action for a specific snake based on epsilon-greedy strategy.
        """
        movement: list[int] = [0] * 4
        if random.random() < snake_data["epsilon"]:
            move: int = random.randrange(4)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            q_values = snake_data["model"](state_tensor)
            move = int(torch.argmax(q_values).item())
        movement[move] = 1
        return movement

    def train_step(self, snake_data, batch) -> None:
        """
        Performs a training step for a specific snake using a batch of experiences.
        """
        states, actions, rewards, next_states, dones = batch

        # Convert inputs to tensors
        state = torch.tensor(states, dtype=torch.float)
        next_state = torch.tensor(next_states, dtype=torch.float)
        action = torch.tensor(actions, dtype=torch.long)
        reward = torch.tensor(rewards, dtype=torch.float)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            dones = (dones,)

        predictions = snake_data["model"](state)
        targets_full = predictions.clone()

        for idx in range(len(dones)):
            q_new = reward[idx]
            if not dones[idx]:
                q_new = reward[idx] + self.gamma * torch.max(
                    snake_data["goal_model"](next_state[idx])
                )

            targets_full[idx][torch.argmax(action[idx]).item()] = q_new

        snake_data["model"].optimizer.zero_grad()
        loss = snake_data["model"].criterion(targets_full, predictions)
        loss.backward()
        snake_data["model"].optimizer.step()

    def replay(self, snake_data) -> None:
        """
        Samples a batch from the memory of a specific snake and trains its model.
        """
        if len(snake_data["memory"]) < self.batch_size:
            return
        batch = random.sample(snake_data["memory"], self.batch_size)
        self.train_step(snake_data, zip(*batch))

    def decay_epsilon(self, snake_data) -> None:
        """
        Decays epsilon for a specific snake after each episode.
        """
        snake_data["epsilon"] = max(self.epsilon_min, snake_data["epsilon"] * self.decay)

    def train(self) -> None:
        """
        Trains the agent using the replay buffers of each snake.
        """
        best_score: int = 0
        best_time: float = 0

        for epoch in range(self.epochs):
            self.interpreter.reset()
            total_reward: float = 0
            start_time: float = time.time()

            while True:
                states = self.interpreter.board.snake_states
                actions = [self.act(state, snake_data) for state, snake_data in zip(states, self.snakes_data)]
                rewards, dones, scores = self.interpreter.board.step(actions)
                next_states = self.interpreter.board.snake_states

                for i, snake in enumerate(self.interpreter.board.snakes):
                    snake_data = self.snakes_data[i]
                    self.remember(snake_data, states[i], actions[i], rewards[i], next_states[i], dones[i])
                    if len(snake_data["memory"]) >= self.batch_size:
                        self.replay(snake_data)

                total_reward += sum(rewards)
                for snake in self.interpreter.board.snakes:
                    if not snake.alive:
                        self.interpreter.board.snakes.remove(snake)
                self.interpreter.render()
                if all(dones):
                    for snake_data in self.snakes_data:
                        self.decay_epsilon(snake_data)
                    for score in scores:
                        if score > best_score:
                            best_score = score
                    if time.time() - start_time > best_time:
                        best_time = time.time() - start_time
                    print(
                        f"Epoch {epoch + 1}/{self.epochs}, "
                        f"Score: {scores}, "
                        f"Best Score: {best_score}, "
                        f"Best Time: {best_time:.2f}s"
                    )
                    break
                # time.sleep(.1)

            duration = time.time() - start_time
            print(f"Epoch {epoch + 1}/{self.epochs}, Reward: {total_reward}, Time: {duration:.2f}s")


if __name__ == "__main__":
    textures = TextureLoader(config)
    interpreter = Interpreter(config, textures)
    plotter = Plotter()
    agent = Agent(
        config=config,
        interpreter=interpreter,
        plotter=plotter,
    )
    agent.train()
