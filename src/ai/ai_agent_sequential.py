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
from src.utils.plotter import Plotter


class AIAgent:
    def __init__(self, config: Config, input_size: int, action_size: int, model_path: str = None):
        """
        Initializes the AI agent for Snake.
        Args:
            config (Config): Configuration settings.
            input_size (int): Input features size.
            action_size (int): Number of actions.
            model_path (str, optional): Path to load pre-trained model.
        """
        self.config = config
        self.input_size = input_size
        self.action_size = action_size
        self.gamma = config.neural_network.training.gamma
        self.batch_size = config.neural_network.training.batch_size
        self.memory = deque(maxlen=100_000)

        # Exploration parameters
        self.epsilon = config.neural_network.training.exploration.epsilon
        self.decay = config.neural_network.training.exploration.decay
        self.epsilon_min = config.neural_network.training.exploration.minimum_rate

        # Models
        self.model = DQNSnake(input_size, action_size)
        self.target_model = DQNSnake(input_size, action_size)
        if model_path:
            self.model.load(model_path)
        self.update_target_model()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.neural_network.training.learning_rate
        )
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        """
        Updates the target model to have the
        same weights as the current model.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done) -> None:
        """
        Stores an experience in the replay buffer.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state) -> list[int]:
        """
        Chooses an action based on epsilon-greedy strategy.
        """
        movement = [0] * 4
        self.decay_epsilon()
        if random.random() < self.epsilon:
            move = random.randrange(self.action_size)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            q_values = self.model(state_tensor)
            move = int(torch.argmax(q_values).item())
        movement[move] = 1
        return movement

    def train_step(self, batch) -> None:
        """
        Performs a training step using a batch of experiences.
        """
        states, actions, rewards, next_states, dones = batch

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        # Q-values for the current states
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Q-values for the next states (using target network)
        next_q_values = self.target_model(next_states).max(1)[0]
        # No future rewards if the episode is done
        next_q_values[dones] = 0.0

        # Compute targets
        targets = rewards + self.gamma * next_q_values

        # Compute loss and backpropagate
        self.optimizer.zero_grad()
        loss = self.criterion(q_values, targets)
        loss.backward()
        self.optimizer.step()

    def replay(self, batch_size) -> None:
        """
        Samples a batch from memory and trains the model.
        """
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        self.train_step((states, actions, rewards, next_states, dones))

    def decay_epsilon(self):
        """
        Decays epsilon after each episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)


if __name__ == "__main__":
    textures = GameTextures(config)
    game_manager = GameManager(config, textures)
    plotter = Plotter()
    input_size: int = len(game_manager.get_state(game_manager.board.snakes[0]))
    agent = AIAgent(config=config, input_size=input_size, action_size=4)

    record = 0

    for epoch in range(agent.epochs):
        game_manager.reset()
        total_reward = 0

        while True:
            state = game_manager.get_state(game_manager.board.snakes[0])
            action = agent.act(state)
            reward, done, score = game_manager.step(action)
            next_state = game_manager.get_state(game_manager.board.snakes[0])

            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.batch_size:
                agent.replay(agent.batch_size)

            total_reward += reward
            if done:
                agent.update_target_model()
                plotter.update(epoch + 1, total_reward, score, agent.epsilon)
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

# if __name__ == "__main__":
#     textures = GameTextures(config)
#     game_manager = GameManager(config, textures)
#     num_snakes = len(game_manager.board.snakes)  # Number of snakes in the game

#     # Initialize agents for each snake
#     agents = [
#         AIAgent(
#             config=config,
#             input_size=len(game_manager.get_state(snake)),
#             action_size=4,
#         )
#         for snake in game_manager.board.snakes
#     ]

#     plotter = Plotter()
#     episodes = 100
#     batch_size = 32
#     record = [0] * num_snakes

#     for epoch in range(episodes):
#         game_manager.reset()
#         total_rewards = [0] * num_snakes 

#         while True:
#             states = [
#                 game_manager.get_state(snake)
#                 for snake in game_manager.board.snakes
#             ]

#             actions = [
#                 agent.act(state) for agent, state in zip(agents, states)
#             ]

#             rewards, dones, scores = game_manager.step(actions)

#             next_states = [
#                 game_manager.get_state(snake)
#                 for snake in game_manager.board.snakes
#             ]

#             for i, agent in enumerate(agents):
#                 agent.remember(
#                     states[i], actions[i], rewards[i], next_states[i], dones[i]
#                 )

#             for agent in agents:
#                 if len(agent.memory) >= batch_size:
#                     agent.replay(batch_size)

#             total_rewards = [r + tr for r, tr in zip(rewards, total_rewards)]
#             if all(dones):
#                 for i, agent in enumerate(agents):
#                     agent.update_target_model()

#                 plotter.update(
#                     epoch + 1, total_rewards, scores, [agent.epsilon for agent in agents]
#                 )

#                 for i, score in enumerate(scores):
#                     print(
#                         f"Snake {i + 1} - Episode {epoch + 1}/{episodes}, "
#                         f"Total Reward: {total_rewards[i]} "
#                         f"Score: {score}, Record: {record[i]}"
#                     )
#                     if score > record[i]:
#                         record[i] = score
#                         print(f"Snake {i + 1}: New high score: {record[i]}")

#                 break

#         time.sleep(0.1)
