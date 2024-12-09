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
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the network.
        """
        return self.model(x)

    def save(self, path: str) -> None:
        """
        Saves the model to a specified file.
        Args:
            path (str): Path to save the model.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Loads the model weights from a specified file.
        Args:
            path (str): Path to the saved model file.
        """
        if Path(path).exists():
            self.load_state_dict(torch.load(path))
        else:
            raise FileNotFoundError(f"Model file not found at {path}")
