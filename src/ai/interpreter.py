from pathlib import Path
import time
import random
from typing import Optional, List
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
            env (Environment): Environment for the agent.
            plotter (Plotter): Plotter for visualizing training.
            gui (GUI): GUI for rendering the game.
            model_path (str, optional): Path to load pre-trained model.
        """
        self.config: Config = config
        self.gui: PygameGUI | None = gui
        self.env: Environment = env
        self.plotter: Plotter = plotter
        # <-- Hyperparameters -->
        self.gamma: float = config.nn.training.gamma
        self.batch_size: int = config.nn.training.batch_size
        self.epochs: int = config.nn.training.epochs
        self.decay: float = config.nn.training.exploration.decay
        self.epsilon = config.nn.training.exploration.epsilon
        self.epsilon_min: float = config.nn.training.exploration.epsilon_min
        # <-- Models -->
        self.shared_model = DeepQSnakeAgent(env.snake_state_size, 4, config)
        self.target_model = DeepQSnakeAgent(env.snake_state_size, 4, config)
        self.memory = deque(maxlen=100_000)
        # Load pre-trained model if provided
        if model_path:
            self.shared_model.load(model_path)
            self.target_model.load_state_dict(self.shared_model.state_dict())

    # <-- Model Utils -->
    def update_target_model(self) -> None:
        """Updates the target model from the shared model."""
        self.target_model.load_state_dict(self.shared_model.state_dict())

    def move_to_device(self, batch: List[torch.Tensor]) -> List[torch.Tensor]:
        """Moves a batch of tensors to the device."""
        return [data.to(self.shared_model.device) for data in batch]

    def cache(self, state, action, reward, next_state, done) -> None:
        """Stores an experience in the shared replay cache."""
        self.memory.append((state, action, reward, next_state, done))

    # <-- Training -->
    def select_action_epsilon_greedy(self, state: torch.Tensor):
        """Chooses an action based on epsilon-greedy strategy."""
        if random.random() >= self.epsilon:
            return self.act(state)
        return random.randrange(4)

    def act(self, state: torch.Tensor):
        """Chooses an action using the shared model."""
        state_tensor: torch.Tensor = state.unsqueeze(0)
        with torch.no_grad():
            q_values = self.shared_model(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def train_step(self, batch) -> None:
        """Training step using a batch of experiences from cache."""
        states, actions, rewards, dones, next_states = self.move_to_device(
            batch
        )

        # Scale the rewards(clip to avoid exploding gradients)
        rewards = torch.clamp(rewards, -1, 1)

        # Move the tensors to the device
        # rewards: torch.Tensor = rewards.to(self.shared_model.device)
        # dones: torch.Tensor = dones.to(self.shared_model.device)

        # Get the Q-values for the current states
        # predictions: torch.Tensor = self.shared_model(states.to(self.shared_model.device))
        predictions: torch.Tensor = self.shared_model(states)

        # Clone the predictions to create the target values
        # Detach basically means don't calculate the gradients for the target values
        targets_full: torch.Tensor = (
            predictions.clone().detach().requires_grad_(False)
        )

        # Batch process next states for efficiency
        with torch.no_grad():
            # q_next = self.target_model(next_states.to(self.target_model.device)).max(dim=1).values
            q_next = self.target_model(next_states).max(dim=1).values

        # Compute Q-values for the terminal states and non-terminal states
        q_new = rewards + (1 - dones) * self.gamma * q_next

        # Update the targets for the actions taken
        # targets_full[torch.arange(self.batch_size), actions] = q_new
        targets_full[range(self.batch_size), actions] = q_new

        # Compute the loss and backpropagate
        loss = self.shared_model.criterion(predictions, targets_full)

        # Backpropagate the loss
        self.shared_model.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to avoid exploding gradients
        # torch.nn.utils.clip_grad_norm_(self.shared_model.parameters(), max_norm=1)

        # Update the weights
        self.shared_model.optimizer.step()

    def replay(self) -> None:
        """Samples and trains a batch from the shared memory."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, dones, next_states = zip(*batch)

        actions = torch.tensor(actions, dtype=torch.long)
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)

        dataset = TensorDataset(states, actions, rewards, dones, next_states)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        for batch in dataloader:
            self.train_step(batch)

    def decay_epsilon(self) -> None:
        """Decays epsilon after each episode."""
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
                states: List[torch.Tensor] = self.env.snake_states
                actions: List[int] = [
                    self.select_action_epsilon_greedy(state)
                    for state in states
                ]
                rewards, dones, scores = self.env.step(actions)
                next_states: List[torch.Tensor] = self.env.snake_states

                for i in range(len(self.env.snakes)):
                    self.cache(
                        states[i],
                        actions[i],
                        rewards[i],
                        dones[i],
                        next_states[i],
                    )
                if len(self.memory) > self.batch_size:
                    self.replay()

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
    gui = None
    # gui = PygameGUI(config)
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
    # try:
    interpreter.train()
    interpreter.evaluate()
    # except KeyboardInterrupt:
    #     pass
    # except Exception as e:
    #     print(e)
