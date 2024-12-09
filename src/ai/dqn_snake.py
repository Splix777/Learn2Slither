from typing import Optional

import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


class DQNSnake(nn.Module):
    def __init__(self, input_size: int, output_size: int, scaler: Optional[MinMaxScaler]):
        """
        A deep Q-network for the Snake AI.

        Args:
            input_size (int): The size of the input state.
            output_size (int): The number of possible actions.
            scaler (MinMaxScaler): Scaler to scaler states.
        """
        super().__init__()
        self.scaler = scaler or MinMaxScaler(feature_range=(0, 1))
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def fit_scaler(self, sample_state: List[int]) -> None:
        """
        Fit the scaler using sample state.

        Args:
            sample_state (list[int]): Sample path to
                fit the scaler.
        """
        self.scaler.fit(sample_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the network.

        Args:
            x (Tensor): Game state of snake.
        """
        if self.scaler:
            x = torch.tensor(
                self.scaler.transform(x.numpy()),
                dtype=torch.float
            )

        return self.model(x)

    def save(self, path: str) -> None:
        """
        Save the model and scaler to a specified file.
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'scaler': self.scaler,
        }, path)

    def load(self, path: str) -> None:
        """
        Loads the model weights from a specified file.

        Args:
            path (str): Path to the saved model file.
        """
        if Path(path).exists():
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler']
        else:
            raise FileNotFoundError(f"Model file not found at {path}")
