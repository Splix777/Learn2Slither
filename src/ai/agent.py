"""Deep Q-Network for the Snake AI."""

import torch
import torch.nn as nn
from pathlib import Path

from src.config.settings import Config


class DeepQSnakeAgent(nn.Module):
    def __init__(self, input_size: int, output_size: int, config: Config):
        """
        A deep Q-network for the Snake AI.

        Args:
            input_size (int): The size of the input state.
            output_size (int): The number of possible actions.
            config (Config): The configuration settings.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.01),
            # nn.Linear(128, 64),
            # nn.LeakyReLU(0.01),
            nn.Linear(128, output_size),
        )
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.nn.training.learning_rate,
            amsgrad=True,
            weight_decay=1e-5
        )
        # self.criterion = nn.MSELoss()
        self.criterion = nn.SmoothL1Loss()
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() 
            else "cpu"
        )
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the network.

        Args:
            x (torch.Tensor): A batch of game states for the Snake game.

        Returns:
            torch.Tensor: Output tensor representing Q-values
                for each action.
        """
        return self.model(x.to(self.device))

    def save(self, path: str | Path) -> None:
        """
        Save the model's state dictionary to a specified file.

        Args:
            path (Union[str, Path]): The file path where the model
                will be saved.
        """
        path = Path(path)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
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
            self.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
            self.to(self.device)
            print(f"Model loaded from {path}")

        except Exception as e:
            raise RuntimeError(f"Error loading model from {path}: {e}") from e

"""
What is Q-Learning?

Q-learning is a decision-making framework where an agent learns to
choose the best actions to maximize rewards over time. The goal is to
build a "map" of the best actions for every situation (state).

The key concept in Q-learning is the Q-value:
    - Q(state, action) is a value that represents the "quality" of
        taking a specific action in a specific state.
    - The agent uses these Q-values to make decisions, aiming to pick
        actions with the highest Q-value.

The Q-value is updated using the Bellman equation:
    Q(s, a) = r + γ * max(Q(s', a'))
    - Q(s, a) is the Q-value for state s and action a.
    - r is the immediate reward for taking action a in state s.
    - γ is the discount factor for future rewards.
    - max(Q(s', a')) is the maximum Q-value for the next state s' and
        all possible actions a'.

----
        
What is a Q-Network?

A Q-network is a neural network that approximates the Q-table when the
problem is too large or complex to store every possible state-action
pair explicitly.

We use a deep Q-network (DQN) to approximate the Q-values for the Snake
game. The network takes the game state as input and outputs Q-values for
each possible action. The agent uses these Q-values to choose the best
action to take.

Analogy:

Imagine the city becomes so large that you cant memorize every street
and its reward. Instead, you learn to recognize patterns:
    - Busy streets during rush hour are slow (low Q-value).
    - Shortcuts throught quiet neighborhoods are fast (high Q-value).
    - Certain landmarks mean a fast route is ahead (high Q-value).

You replace your mental map with a neural network that learns these
patterns from experience. The network takes the current location as
input and outputs the best direction to take.

----

How is a Q-Network Different from a Regular Neural Network?

Supervised vs Reinforcement Learning

Supervised Learning:
    Regular neural networks learn from labeled data. The network
    adjusts its weights to minimize the difference between its
    predictions and the true labels.

Reinforcement Learning:
    Q-networks learn from rewards and punishments. The network
    adjusts its weights to maximize the expected rewards over time.


TL:DR
Q-Networks serve as 'smart-maps' for agents to navigate complex
environments. They approximate Q-values to make decisions that
maximize rewards over time.
"""