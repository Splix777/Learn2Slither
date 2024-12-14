from pathlib import Path
import time
import random
from typing import Optional, List, Tuple
from collections import deque

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.ui.pygame.pygame_gui import PygameGUI
from src.utils.plotter import Plotter
from src.ai.agent import DeepQSnakeAgent
from src.game.environment import Environment
from src.config.settings import Config, config


class Interpreter:
    def __init__(
        self,
        config: Config,
        env: Environment,
        plotter: Plotter,
        gui: Optional[PygameGUI] = None,
        model_path: Optional[Path] = None,
    ) -> None:
        """
        Initializes the AI agent for Snake.
        Args:
            config (Config): Configuration settings.
            env (Environment): Environment for the agent.
            plotter (Plotter): Plotter for visualizing training.
            gui (GUI): GUI for rendering the game.
            model_path (str, optional): Path to load pre-trained model.
        """
        self.config: Config = config
        self.gui: PygameGUI | None = gui
        self.env: Environment = env
        self.plotter: Plotter = plotter
        # <-- Hyperparameters -->
        self.gamma: float = config.nn.training.gamma
        self.batch_size: int = config.nn.training.batch_size
        self.epochs: int = config.nn.training.epochs
        self.decay: float = config.nn.training.exploration.decay
        self.epsilon = config.nn.training.exploration.epsilon
        self.epsilon_min: float = config.nn.training.exploration.epsilon_min
        # <-- Models -->
        self.train_model = DeepQSnakeAgent(env.snake_state_size, 4, config)
        self.target_model = DeepQSnakeAgent(env.snake_state_size, 4, config)
        self.memory = deque(maxlen=100_000)
        # Load pre-trained model if provided
        if model_path:
            self.train_model.load(model_path)
            self.target_model.load_state_dict(self.train_model.state_dict())

    # <-- Model Utils -->
    def update_target_model(self) -> None:
        """Updates the target model from the shared model."""
        self.target_model.load_state_dict(self.train_model.state_dict())

    def move_to_device(self, batch: List[torch.Tensor]) -> List[torch.Tensor]:
        """Moves a batch of tensors to the device."""
        return [data.to(self.train_model.device) for data in batch]

    def cache(self, experiences: List[Tuple]) -> None:
        """Stores an experience in the shared replay cache."""
        self.memory.extend(experiences)

    def decay_epsilon(self) -> None:
        """Decays epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)

    def save_model(self, path: str | Path) -> None:
        """Saves the model to a specified file."""
        self.train_model.save(path)
    # <-- Training -->
    def act(self, state: torch.Tensor):
        """Chooses an action based on epsilon-greedy strategy."""
        if random.random() >= self.epsilon:
            return self.choose_action(state)
        return random.randrange(4)

    def choose_action(self, state: torch.Tensor):
        """Chooses an action using the shared model."""
        state_tensor: torch.Tensor = state.unsqueeze(0)
        with torch.no_grad():
            q_values = self.train_model(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def train_step(self, batch) -> None:
        """Training step using a batch of experiences from cache."""
        state, action, reward, done, next_state = self.move_to_device(batch)

        # Reward clipping
        reward = torch.clamp(reward, -1, 1)

        # Get the Q-values for the current states
        q_values = self.train_model(state)

        # Get the Q-values for the next states
        next_q_values = q_values.clone().detach().requires_grad_(False)

        # Compute the Q-values for the next states
        with torch.no_grad():
            # No gradient calculation for the target model since
            # we are not training it. We just use it to get the
            # Q-values for the next states. That way it does not
            # affect the gradients of the training model.
            q_next = self.target_model(next_state).max(dim=1).values

        # Apply the Bellman equation
        q_new = reward + (1 - done) * self.gamma * q_next

        next_q_values[range(self.batch_size), action] = q_new

        # Compute the loss and backpropagate
        loss = self.train_model.criterion(q_values, next_q_values)

        # Backpropagate the loss
        self.train_model.optimizer.zero_grad()
        loss.backward()

        # Update the weights
        self.train_model.optimizer.step()

    def replay(self) -> None:
        """Samples and trains a batch from the shared memory."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        actions = torch.tensor(actions, dtype=torch.long)
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)

        dataset = TensorDataset(states, actions, rewards, dones, next_states)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        for batch in dataloader:
            self.train_step(batch)

    def gather_experience(self) -> Tuple[List, List, List]:
        """Gathers and zips experiences from the environment."""
        states: List[torch.Tensor] = self.env.snake_states
        actions: List[int] = [self.act(state) for state in states]
        rewards, dones, scores = self.env.step(actions)
        next_states: List[torch.Tensor] = self.env.snake_states

        # Zip the data into experiences
        experiences = list(zip(states, actions, rewards, next_states, dones))
        return experiences, scores, rewards

    def train_epoch(self) -> Tuple[int, float, float]:
        """Runs a single training epoch."""
        best_score: int = 0
        best_time: float = 0
        total_rewards: float = 0
        self.env.reset()
        start_time: float = time.time()

        while True:
            experiences, scores, rewards = self.gather_experience()
            total_rewards += sum(rewards)
            self.cache(experiences)

            if len(self.memory) > self.batch_size:
                self.replay()

            if self.gui:
                self.gui.training_render(self.env.game_state)

            if all(done for _, _, _, _, done in experiences):
                self.decay_epsilon()
                self.replay()

                for score in scores:
                    if score > best_score:
                        best_score = score
                elapsed_time = time.time() - start_time
                if elapsed_time > best_time:
                    best_time = elapsed_time
                break

        return best_score, best_time, total_rewards


    def train(self) -> None:
        """Trains the agent over multiple epochs."""
        best_score: int = 0
        best_time: float = 0

        for epoch in range(self.epochs):
            epoch_score, epoch_time, rewards = self.train_epoch()
            best_score = max(best_score, epoch_score)
            best_time = max(best_time, epoch_time)
            plotter.update(epoch, rewards, epoch_score, best_score, self.epsilon)

            if epoch % config.nn.training.update_frequency == 0:
                self.update_target_model()

        plotter.close()

    @torch.no_grad()
    def evaluate(self, test_episodes: int = 10) -> None:
        """
        Evaluates the agent's performance over multiple episodes without training.
        """
        for i in range(test_episodes):
            self.env.reset()
            total_reward: float = 0

            while True:
                states = self.env.snake_states
                actions = [self.choose_action(state) for state in states]
                rewards, dones, scores = self.env.step(actions)
                total_reward += sum(rewards)

                if self.gui:
                    self.gui.training_render(self.env.game_state)

                if all(dones):
                    print(
                        f"Episode {i + 1}/{test_episodes}, "
                        f"Reward: {total_reward}, "
                        f"Score: {sum(scores)}"
                    )
                    break


if __name__ == "__main__":
    gui = None
    gui = PygameGUI(config)
    env = Environment(config)
    plotter = Plotter()
    saved_model: Optional[Path] = config.paths.models / "snake_best.pth"
    if not saved_model.exists():
        saved_model = None
    interpreter = Interpreter(
        config=config,
        gui=gui or None,
        env=env,
        plotter=plotter,
        # model_path=saved_model or None,
    )
    try:
        interpreter.train()
        interpreter.evaluate()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
