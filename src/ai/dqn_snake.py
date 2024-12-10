from typing import Optional, List

import numpy as np

import torch
import torch.nn as nn
from pathlib import Path


class DQNSnake(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        """
        A deep Q-network for the Snake AI.

        Args:
            input_size (int): The size of the input state.
            output_size (int): The number of possible actions.
            scaler (MinMaxScaler): Scaler to scaler states.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.01),
            # nn.Linear(64, 64),
            # nn.LeakyReLU(0.01),
            nn.Linear(128, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the network.

        Args:
            x (Tensor): Game state of snake.
        """
        return self.model(x)

    def save(self, path: str) -> None:
        """
        Save the model and scaler to a specified file.
        """
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path: str) -> None:
        """
        Loads the model weights from a specified file.

        Args:
            path (str): Path to the saved model file.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
