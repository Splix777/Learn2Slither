from typing import Union
import torch
import torch.nn as nn
from pathlib import Path


class DQNSnake(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        """
        A deep Q-network for the Snake AI.

        Args:
            input_size (int): The size of the input state.
            output_size (int): The number of possible actions.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, output_size)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the network.

        Args:
            x (torch.Tensor): A batch of game states for the Snake game.

        Returns:
            torch.Tensor: Output tensor representing Q-values for each action.
        """
        return self.model(x.to(self.device))

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model's state dictionary to a specified file.

        Args:
            path (Union[str, Path]): The file path where the model will be saved.
        """
        path = Path(path)  # Ensure it's a Path object
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """
        Load the model weights from a specified file.

        Args:
            path (Union[str, Path]): Path to the saved model file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        
        try:
            self.load_state_dict(torch.load(path, map_location=self.device))
            self.to(self.device)
            print(f"Model loaded from {path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model from {path}: {e}")
