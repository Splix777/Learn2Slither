import random
import time
from typing import List
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from src.game.game_manager import GameManager
from src.game.board import GameBoard
from src.game.snake import Snake
from src.ui.game_textures import GameTextures
from src.config.settings import Config, get_config

class DQNSnake(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(DQNSnake, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self.state_mapping = {
            "wall": 0,
            "snake_head": 1,
            "snake_body": 2,
            "green_apple": 3,
            "red_apple": 4,
            "empty": 5
        }

    def forward(self, x: List[int]) -> torch.Tensor:
        """
        Takes a state and returns the Q-values of each action.

        Args:
            x (dict[str, list[str]]): The state of the game.

        Returns:
            torch.Tensor: The Q-values of each action. (left, right, forward)
        """
        processed_state: torch.Tensor = torch.FloatTensor(x).unsqueeze(0)
        return self.model(processed_state)


class AIAgent:
    def __init__(self, config: Config, input_size: int, action_size: int):
        self.config = config
        self.input_size = input_size
        self.action_size = action_size
        self.epsilon = config.neural_network.training.exploration.initial_rate
        self.epsilon_min = (
            config.neural_network.training.exploration.minimum_rate
        )
        self.epsilon_decay = config.neural_network.training.exploration.decay
        self.gamma = config.neural_network.training.discount_factor
        self.learning_rate = config.neural_network.training.learning_rate

        # Neural network and optimizer
        self.model = DQNSnake(input_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        # Replay buffer
        self.memory = deque(maxlen=200_000)

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state) -> int:
        # Exploration vs Exploitation
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return int(x=torch.argmax(input=q_values).item())  # Best action

    def replay(self, batch_size) -> None:
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target_tensor = self.model(state_tensor).clone().detach()

            target_tensor = target_tensor.squeeze(0)

            # Print shape and action for debugging
            # print(f"target_tensor shape: {target_tensor.shape}")
            # print(f"action: {action}")
            
            if action < target_tensor.shape[1]:  # Ensure the action is within bounds
                target_tensor[0][action] = target
            else:
                pass
                # print(f"Warning: Action {action} is out of bounds for target_tensor of size {target_tensor.shape}")
        

            # Train the network
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state_tensor), target_tensor)
            loss.backward()
            self.optimizer.step()

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    config: Config = get_config()
    textures = GameTextures(config)
    game_manager = GameManager(config, textures)
    input_size: int = game_manager.board.snake_vision
    agent = AIAgent(config, input_size=input_size, action_size=4)

    episodes = 1000
    batch_size = 32

    # print(agent.model.forward(game_manager.get_snake_vision(game_manager.snakes[0])))

    for e in range(episodes):
        # Reset the game
        game_manager = GameManager(config, textures)
        state = game_manager.get_snake_vision(game_manager.snakes[0])
        total_reward = 0

        while True:  # Game loop
            action = agent.act(state)
            next_state, reward, done = game_manager.step(action)  # Implement step logic

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}")
                break
            time.sleep(0.2)

        agent.replay(batch_size)
