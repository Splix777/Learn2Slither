import time
from typing import Optional, Tuple

import torch

from src.ui.pygame.pygame_gui import PygameGUI
from src.utils.plotter import Plotter
from src.game.snake import Snake
from src.ai.agent import Brain
from src.game.environment import Environment
from src.config.settings import Config, config


class ReinforcementLearner:
    def __init__(
        self,
        config: Config,
        env: Environment,
        # plotter: Plotter,
        gui: Optional[PygameGUI] = None,
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
        # self.plotter: Plotter = plotter

    # <-- Model Utils -->
    def train_epoch(self) -> Tuple[int, float, float]:
        """Runs a single training epoch."""
        best_score: int = 0
        best_time: float = 0
        total_rewards: float = 0
        self.env.reset()
        start_time: float = time.time()

        while True:
            for snake in self.env.snakes:
                self.env.train_step(snake)
                best_score = max(best_score, snake.size)
                total_rewards += snake.reward

            for snake in self.env.snakes:
                if snake.brain:
                    snake.brain.replay()

            if self.gui:
                self.gui.training_render(self.env.map)

            if all(not snake.alive for snake in self.env.snakes):
                for snake in self.env.snakes:
                    if snake.brain:
                        snake.brain.replay()
                        snake.brain.decay_epsilon()

                elapsed_time: float = time.time() - start_time
                best_time = max(best_time, elapsed_time)
                break
            # time.sleep(0.1)

        return best_score, best_time, total_rewards

    def train(self) -> None:
        """Trains the agent over multiple epochs."""
        best_score: int = 0
        best_time: float = 0
        epsilon: float = self.config.nn.exploration.epsilon

        for epoch in range(self.config.nn.epochs):
            epoch_score, epoch_time, rewards = self.train_epoch()
            best_score = max(best_score, epoch_score)
            best_time = max(best_time, epoch_time)
            for snake in self.env.snakes:
                if snake.brain:
                    epsilon = min(epsilon, snake.brain.epsilon)

            # plotter.update(epoch, rewards, epoch_score, best_score, epsilon)

            if epoch % config.nn.update_frequency == 0:
                for snake in self.env.snakes:
                    if snake.brain:
                        snake.brain.save(
                            config.paths.models / f"snake_{snake.id}.pth"
                        )

        # plotter.close()

    @torch.no_grad()
    def evaluate(self, test_episodes: int = 10) -> None:
        """
        Evaluates the agent's performance over multiple episodes without training.
        """
        for i in range(test_episodes):
            self.env.reset()
            max_score: int = 0

            while True:
                self.env.step()
                max_score = max(max_score, max(snake.size for snake in self.env.snakes))

                if self.gui:
                    self.gui.training_render(self.env.map)

                if all(not snake.alive for snake in self.env.snakes):
                    print(
                        f"Episode {i + 1}/{test_episodes}, "
                        f"Max Score: {max_score}, "
                    )
                    break
                time.sleep(0.2)


if __name__ == "__main__":
    gui = PygameGUI(config)
    # plotter = Plotter()
    model_path = config.paths.models / "snake_0.pth"

    if not model_path.exists():
        model_path = None

    # Shared brain for all snakes
    brain = Brain(config=config, model_path=model_path)

    # Create 10 snakes with brains
    snakes = [Snake(id=i, brain=brain) for i in range(10)]


    env = Environment(config, snakes)

    interpreter = ReinforcementLearner(
        config=config,
        gui=gui or None,
        env=env,
        # plotter=plotter,
    )
    try:
        interpreter.train()
        interpreter.evaluate()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
