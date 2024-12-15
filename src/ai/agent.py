"""Deep Q-Network for the Snake AI."""

from typing import Tuple, Optional
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pathlib import Path
from collections import deque

from src.config.settings import Config


class Brain(nn.Module):
    def __init__(self, config: Config, model_path: Optional[str | Path] = None):
        """A deep Q-network for the Snake AI."""
        super().__init__()
        # <-- Hyperparameters -->
        self.gamma: float = config.nn.gamma
        self.batch_size: int = config.nn.batch_size
        self.epochs: int = config.nn.epochs
        self.decay: float = config.nn.exploration.decay
        self.epsilon = config.nn.exploration.epsilon
        self.epsilon_min: float = config.nn.exploration.epsilon_min
        # <-- Attributes -->
        self.input_size = config.nn.input_shape
        self.output_size = config.nn.output_shape
        self.memory = deque(maxlen=100_000)
        # <-- Model -->
        self.model = nn.Sequential(
            nn.Linear(config.nn.input_shape, 128),
            nn.LeakyReLU(0.01),
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
        # <-- Load Model -->
        if model_path:
            self.load(model_path)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the network."""
        return self.model(x.to(self.device))

    def train_step(self, batch: Tuple) -> None:
        """Training logic for the network."""
        state, action, reward, next_state, done = self.batch_to_device(batch)

        reward = torch.clamp(reward, min=-1, max=1)

        q_values: torch.Tensor = self.model(state)
        next_q_values: torch.Tensor = (
            q_values.clone().detach().requires_grad_(False)
        )

        with torch.no_grad():
            q_next = self.model(next_state).max(dim=1).values

        q_new = reward + (1 - done) * self.gamma * q_next

        if action.ndim == 2:
            action = torch.argmax(action, dim=1)

        next_q_values[range(self.batch_size), action] = q_new

        loss = self.criterion(q_values, next_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def replay(self) -> None:
        """Replay experiences for training."""
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
        """Cache experiences for training."""
        self.memory.append(experiences)

    def act(self, state: torch.Tensor) -> int:
        """Chooses an action based on epsilon-greedy strategy."""
        if random.random() >= self.epsilon:
            return self.choose_action(state)
        return random.randrange(self.output_size)

    def choose_action(self, state: torch.Tensor):
        """Chooses an action using the shared model."""
        state_tensor: torch.Tensor = state.unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(self.tensor_to_device(state_tensor))
        return int(torch.argmax(q_values, dim=1).item())

    def decay_epsilon(self) -> None:
        """Decay exploration rate (AI-specific)."""
        self.epsilon: float = max(self.epsilon_min, self.epsilon * self.decay)

    def batch_to_device(self, batch: Tuple):
        """Moves a batch of tensors to the device."""
        return (tensor.to(self.device) for tensor in batch)

    def tensor_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Moves a tensor to the device."""
        return tensor.to(self.device)

    def save(self, path: str | Path) -> None:
        """Save the model's state dictionary to a specified file."""
        path = Path(path)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        """Load the model weights from a specified file."""
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
