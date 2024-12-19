"""Agent module for the Snake AI."""

from typing import Tuple, Optional, Generator
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pathlib import Path
from collections import deque

from src.config.settings import Config


class Agent(nn.Module):
    def __init__(self, config: Config, path: Optional[str | Path] = None):
        super().__init__()
        self.gamma = config.nn.gamma
        self.batch_size = config.nn.batch_size
        self.epochs = config.nn.epochs
        self.decay = config.nn.exploration.decay
        self.epsilon = config.nn.exploration.epsilon
        self.epsilon_min = config.nn.exploration.epsilon_min
        self.loss_value = float("inf")

        self.input_size = config.nn.input_shape
        self.output_size = config.nn.output_shape
        self.memory = deque(maxlen=config.nn.memory_size)

        self.model = nn.Sequential(
            nn.Linear(config.nn.input_shape, 128),
            nn.LeakyReLU(0.01),
            # nn.Linear(128, 64),
            # nn.Dropout(0.1),
            nn.Linear(128, config.nn.output_shape),
        )
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.nn.learning_rate,
            amsgrad=True,
            weight_decay=1e-5,
        )
        self.criterion = nn.SmoothL1Loss(beta=0.8)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

        if path:
            self.load(path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network."""
        return self.model(x.to(self.device))

    def train_step(self, batch: Tuple) -> None:
        """Single training step for the neural network."""
        state, action, reward, next_state, done = self.batch_to_device(batch)
        reward = torch.clamp(reward, min=-1, max=1)
        current_q_values: torch.Tensor = self.model(state)

        with torch.no_grad():
            next_q_values = self.model(next_state)
            max_next_q_values, _ = torch.max(next_q_values, dim=1)

        target_q_values = reward + (1 - done) * self.gamma * max_next_q_values

        if action.ndim == 2:
            action = torch.argmax(action, dim=1)

        q_values_for_actions = current_q_values.gather(
            1, action.unsqueeze(1)
        ).squeeze(1)

        loss: torch.Tensor = self.criterion(
            q_values_for_actions, target_q_values
        )
        self.loss_value = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()

    def replay(self) -> None:
        """Replay experiences from memory."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        actions = torch.tensor(actions, dtype=torch.long)
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.cat(rewards)
        dones = torch.cat(dones)

        dataset = TensorDataset(states, actions, rewards, next_states, dones)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        for batch in dataloader:
            self.train_step(batch)

    def cache(self, experiences: Tuple) -> None:
        """Cache experiences for replay."""
        self.memory.append(experiences)

    def act(self, state: torch.Tensor, learn: bool) -> int:
        """Choose an action based on epsilon-greedy policy."""
        # self.epsilon = self.epsilon if learn else 0.005
        if random.random() <= self.epsilon and learn:
            return random.randrange(self.output_size)
        return self.choose_action(state)

    def choose_action(self, state: torch.Tensor) -> int:
        """Choose the best action based on Q-values."""
        with torch.no_grad():
            q_values = self.model(state.unsqueeze(0).to(self.device))
        return int(torch.argmax(q_values, dim=1).item())

    def decay_epsilon(self) -> None:
        """Decay the exploration rate over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)

    def batch_to_device(
        self, batch: Tuple[torch.Tensor, ...]
    ) -> Generator[torch.Tensor, None, None]:
        """Move batch to the correct device."""
        return (tensor.to(self.device) for tensor in batch)

    def tensor_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the correct device."""
        return tensor.to(self.device)

    def save(self, path: str | Path) -> None:
        """Save the model to a file."""
        path = Path(path)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        """Load the model from a file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")

        try:
            self.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
            self.to(self.device)
            print(f"Model loaded from {path}")

        except Exception as e:
            raise RuntimeError(f"Error loading model from {path}: {e}") from e
