import random
import time
from typing import Optional
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from src.game.interpreter import Interpreter
from src.ui.texture_loader import TextureLoader
from src.config.settings import Config, config
from src.utils.plotter import Plotter
from src.ai.dqn_snake import DQNSnake


class Agent:
    def __init__(self, config: Config, input_size: int, action_size: int, model_path: Optional[str] = None):
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
        self.epochs = config.neural_network.training.epochs
        self.memory = deque(maxlen=100_000)

        # Exploration parameters
        self.epsilon = config.neural_network.training.exploration.epsilon
        self.decay = config.neural_network.training.exploration.decay
        self.epsilon_min = config.neural_network.training.exploration.epsilon_min

        # Models
        self.model = DQNSnake(input_size, action_size)
        self.target_model = DQNSnake(input_size, action_size)
        if model_path:
            self.model.load(model_path)
        self.update_target_model()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.neural_network.training.learning_rate,
            weight_decay=1e-5
        )
        self.criterion = nn.MSELoss()

    def update_target_model(self) -> None:
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

        # Convert inputs to tensors
        state = torch.tensor(states, dtype=torch.float)
        next_state = torch.tensor(next_states, dtype=torch.float)
        action = torch.tensor(actions, dtype=torch.long)
        reward = torch.tensor(rewards, dtype=torch.float)

        # Add batch dimension if necessary
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            dones = (dones,)

        # Get predictions for current state
        predictions = self.model(state)
        targets_full = predictions.clone()

        for idx in range(len(dones)):
            q_new = reward[idx]
            if not dones[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            targets_full[idx][torch.argmax(action[idx]).item()] = q_new

        # Compute loss
        self.optimizer.zero_grad()
        loss = self.criterion(targets_full, predictions)
        loss.backward()
        self.optimizer.step()

    def replay(self, batch_size) -> None:
        """
        Samples a batch from memory and trains the model.
        """
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        self.train_step(zip(*batch))

    def decay_epsilon(self) -> None:
        """
        Decays epsilon after each episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)


if __name__ == "__main__":
    textures = TextureLoader(config)
    interpreter = Interpreter(config, textures)
    plotter = Plotter()
    input_size: int = len(interpreter.get_state(interpreter.board.snakes[0]))
    agent = Agent(config=config, input_size=input_size, action_size=4)

    record = 0
    record_time = 0

    for epoch in range(agent.epochs):
        interpreter.reset()
        total_reward = 0
        start_time = time.time()

        while True:
            state = interpreter.get_state(interpreter.board.snakes[0])
            action = agent.act(state)
            reward, done, score = interpreter.step(action)
            next_state = interpreter.get_state(interpreter.board.snakes[0])

            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.batch_size:
                agent.replay(agent.batch_size)

            total_reward += reward
            if done:
                longest_time = time.time() - start_time
                agent.update_target_model()
                # plotter.update(epoch + 1, total_reward, score, agent.epsilon)
                print(
                    f"Episode {epoch + 1}/{agent.epochs}, "
                    f"Total Reward: {total_reward} "
                    f"Score: {score}, Record: {record} "
                    f"Time: {longest_time:.2f}, Record Time {record_time:.2f}"
                )
                if score > record:
                    record = score
                    print(f"New high score: {record}")
                if longest_time > record_time:
                    record_time = longest_time
                    print(f"New longest time: {record_time}")
                break

        time.sleep(0.1)
