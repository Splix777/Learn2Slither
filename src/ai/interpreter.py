from pathlib import Path
import time
import random
from typing import Optional
from collections import deque

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.ui.pygame.pygame_gui import PygameGUI
from src.utils.plotter import Plotter
from src.ai.agent import DeepQSnakeAgent
from src.game.environment import Environment
from src.config.settings import Config, config


class Interpreter:
    def __init__(
        self,
        config: Config,
        env: Environment,
        plotter: Plotter,
        gui: Optional[PygameGUI] = None,
        model_path: Optional[Path] = None,
    ) -> None:
        """
        Initializes the AI agent for Snake.
        Args:
            config (Config): Configuration settings.
            gui (GUI): GUI for rendering the game.
            plotter (Plotter): Plotter for visualizing training.
            model_path (str, optional): Path to load pre-trained model.
        """
        self.config: Config = config
        self.gui: PygameGUI | None = gui
        self.env: Environment = env
        self.plotter: Plotter = plotter
        self.gamma: float = config.nn.training.gamma
        self.batch_size: int = config.nn.training.batch_size
        self.epochs: int = config.nn.training.epochs
        self.decay: float = config.nn.training.exploration.decay
        self.epsilon = config.nn.training.exploration.epsilon
        self.epsilon_min: float = config.nn.training.exploration.epsilon_min

        # Initialize a single model and a target model for training stability
        self.shared_model = DeepQSnakeAgent(env.snake_state_size, 4, config)
        self.target_model = DeepQSnakeAgent(env.snake_state_size, 4, config)
        self.memory = deque(maxlen=100_000)

        # Load pre-trained model if provided
        if model_path:
            self.shared_model.load(model_path)
            self.target_model.load_state_dict(self.shared_model.state_dict())

    def update_target_model(self) -> None:
        """
        Updates the target model by copying weights from the shared model.
        """
        self.target_model.load_state_dict(self.shared_model.state_dict())

    def cache(self, state, action, reward, next_state, done) -> None:
        """
        Stores an experience in the shared replay buffer.
        """
        self.memory.append((state, action, reward, next_state, done))

    def exploration_act(self, state: torch.Tensor) -> list[int]:
        """
        Chooses an action based on epsilon-greedy strategy using the shared model.
        """
        if random.random() >= self.epsilon:
            return self.act(state)
        movement = torch.zeros(4, dtype=torch.float)
        movement[random.randrange(4)] = 1
        return movement.int().tolist()

    def act(self, state: torch.Tensor) -> list[int]:
        """
        Chooses an action based on the greedy strategy using the shared model.
        """
        # state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        state_tensor: torch.Tensor = state.unsqueeze(0)
        with torch.no_grad():
            q_values = self.shared_model(state_tensor)
        move = torch.argmax(q_values, dim=1).item()
        movement = torch.zeros(4)
        movement[int(move)] = 1
        return movement.int().tolist()

    def train_step(self, batch) -> None:
        """
        Performs a training step using a batch of experiences from the shared memory.
        """
        states, actions, rewards, next_states, dones = batch

        predictions: torch.Tensor = self.shared_model(states)
        targets_full: torch.Tensor = predictions.clone().detach() 

        for idx in range(len(dones)):
            q_new = rewards[idx]
            if not dones[idx]:
                q_new = rewards[idx] + self.gamma * torch.max(self.target_model(next_states[idx]))

            targets_full[idx][actions[idx]] = q_new

        self.shared_model.optimizer.zero_grad()
        loss = self.shared_model.criterion(targets_full, predictions)
        loss.backward()
        self.shared_model.optimizer.step()

    def replay(self) -> None:
        """
        Samples a batch from the shared memory and trains the model.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Create a DataLoader from the memory, batching the data
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states: torch.Tensor = torch.stack(list(states))
        actions: torch.Tensor = torch.tensor(actions, dtype=torch.long)
        rewards: torch.Tensor = torch.tensor(rewards, dtype=torch.float)
        next_states: torch.Tensor = torch.stack(list(next_states))
        dones: torch.Tensor = torch.tensor(dones, dtype=torch.bool)
        
        # Use TensorDataset and DataLoader for batching
        dataset = TensorDataset(states, actions, rewards, next_states, dones)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for batch in dataloader:
            self.train_step(batch)

    def decay_epsilon(self) -> None:
        """
        Decays epsilon after each episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)

    def train(self) -> None:
        """
        Trains the agent using the shared replay buffer.
        """
        best_score: int = 0
        best_time: float = 0

        for epoch in range(self.epochs):
            self.env.reset()
            total_reward: float = 0
            start_time: float = time.time()

            while True:
                states = self.env.snake_states
                actions = [self.exploration_act(state) for state in states]
                rewards, dones, scores = self.env.step(actions)
                next_states = self.env.snake_states

                for i in range(len(self.env.snakes)):
                    self.cache(
                        states[i],
                        actions[i],
                        rewards[i],
                        next_states[i],
                        dones[i],
                    )
                if len(self.memory) > self.batch_size:
                    self.replay()

                total_reward += sum(rewards)

                if self.gui:
                    self.gui.training_render(self.env.game_state)

                if all(dones):
                    self.decay_epsilon()
                    for score in scores:
                        if score > best_score:
                            self.update_target_model()
                            if (
                                config.paths.models / "snake_best.pth"
                            ).exists():
                                (
                                    config.paths.models / "snake_best.pth"
                                ).unlink()
                            self.target_model.save(
                                config.paths.models / "snake_best.pth"
                            )
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

            if epoch % config.nn.training.update_frequency == 0:
                self.update_target_model()

            duration = time.time() - start_time
            print(
                f"Epoch {epoch + 1}/{self.epochs}, Reward: {total_reward}, Time: {duration:.2f}s"
            )

    @torch.no_grad()
    def evaluate(self, test_episodes: int = 10) -> None:
        """
        Evaluates the agent's performance over multiple episodes without training.
        """
        for i in range(test_episodes):
            self.env.reset()
            total_reward: float = 0

            while True:
                states = self.env.snake_states
                actions = [self.act(state) for state in states]
                rewards, dones, scores = self.env.step(actions)
                total_reward += sum(rewards)

                if self.gui:
                    self.gui.training_render(self.env.game_state)

                if all(dones):
                    print(
                        f"Episode {i + 1}/{test_episodes}, "
                        f"Reward: {total_reward}, "
                        f"Score: {sum(scores)}"
                    )
                    break


if __name__ == "__main__":
    # gui = PygameGUI(config)
    gui = None
    env = Environment(config)
    plotter = Plotter()
    saved_model: Optional[Path] = config.paths.models / "snake_best.pth"
    if not saved_model.exists():
        saved_model = None
    interpreter = Interpreter(
        config=config,
        gui=gui or None,
        env=env,
        plotter=plotter,
        model_path=saved_model or None,
    )
    try:
        interpreter.train()
        # interpreter.evaluate()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
