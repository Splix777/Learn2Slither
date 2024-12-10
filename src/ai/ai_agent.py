import random
import time
from typing import List
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.game.game_manager import GameManager
from src.ui.game_textures import GameTextures
from src.config.settings import Config, config

import logging
logging.basicConfig(level=logging.INFO, filename="ai_agent.log")

class DQNSnake(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        # super(DQNSnake, self).__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(input_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(256, output_size)
        # )
        super().__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256, output_size)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes a state and returns the Q-values of each action.

        Args:
            x (torch.Tensor): The state of the game.

        Returns:
            torch.Tensor: The Q-values of each action.
                (up, down, left, right)
        """
        x = F.relu(self.linear1(x))
        return self.linear2(x)


        # return self.model(x)


class AIAgent:
    def __init__(self, config: Config, input_size: int, action_size: int):
        self.config = config
        self.input_size = input_size
        self.action_size = action_size
        self.epochs = config.neural_network.training.epochs
        self.epsilon = config.neural_network.training.exploration.epsilon
        self.epsilon_min = config.neural_network.training.exploration.epsilon_min
        self.epsilon_decay = config.neural_network.training.exploration.decay
        self.gamma = config.neural_network.training.gamma
        self.learning_rate = config.neural_network.training.learning_rate
        # <-- Neural network and optimizer -->
        self.model = DQNSnake(input_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        # <-- Replay buffer -->
        self.memory = deque(maxlen=100_000)

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state) -> list[int]:
        movement = [0, 0, 0, 0]
        # Exploration vs Exploitation
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        logging.info(f"Epsilon: {self.epsilon}")
        if random.random() < self.epsilon:
            # Random Action
            move = random.randrange(self.action_size)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            q_values = self.model(state_tensor)
            # Best 
            move = int(torch.argmax(input=q_values).item())
        
        movement[move] = 1
        return movement

    def train_step(self, state, action, reward, next_state, done) -> None:
        # Convert inputs to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Add batch dimension if necessary
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # Get predictions for current state
        predictions = self.model(state)
        targets_full = predictions.clone()

        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            targets_full[idx][torch.argmax(action[idx]).item()] = q_new

        # # Compute targets
        # next_state_values = torch.max(self.model(next_state), dim=1)[0]
        # # Set next_state_values to 0 where done is True
        # next_state_values = next_state_values * (~torch.tensor(done, dtype=torch.bool)).float()

        # targets = reward + self.gamma * next_state_values
        # # Use gather to set the target for the taken action
        # targets_full = predictions.clone()  # Clone predictions to modify targets
        # targets_full.gather(1, action.unsqueeze(1))[:] = targets

        # Compute loss
        self.optimizer.zero_grad()
        loss = self.criterion(targets_full, predictions)
        loss.backward()
        self.optimizer.step()

    def replay(self, batch_size) -> None:
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)

        state, action, reward, next_state, done = zip(*batch)
        self.train_step(state, action, reward, next_state, done)

if __name__ == "__main__":
    textures = GameTextures(config)
    game_manager = GameManager(config, textures)
    input_size: int = len(game_manager.get_state(game_manager.board.snakes[0]))
    agent = AIAgent(config, input_size=input_size, action_size=4)

    episodes = 100
    batch_size = 32
    record = 0

    # game_manager.step(0)
    # print(game_manager.get_state(game_manager.board.snakes[0]))

    for epoch in range(agent.epochs):
        game_manager.reset()
        total_reward = 0

        while True:
            state = game_manager.get_state(
                game_manager.board.snakes[0]
            )
            logging.info(f"State: {state}")

            action: list[int] = agent.act(state)

            reward, done, score = game_manager.step(action)
            logging.info(f"Action: {action}, Reward: {reward}, Done: {done}, Score: {score}")
            
            next_state = game_manager.get_state(
                game_manager.board.snakes[0]
            )

            agent.train_step(state, action, reward, next_state, done)

            agent.remember(state, action, reward, next_state, done)

            total_reward += reward

            if done:
                print(
                    f"Episode {epoch + 1}/{agent.epochs}, "
                    f"Total Reward: {total_reward} "
                    f"Score: {score}, Record: {record}"
                )
                if score > record:
                    record = score
                    print(f"New high score: {record}")
                break
            time.sleep(0.1)

        agent.replay(batch_size)
