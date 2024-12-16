import time
from typing import Optional, Tuple

import torch
import rich
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

from src.ui.pygame.pygame_gui import PygameGUI
from src.utils.plotter import Plotter
from src.game.snake import Snake
from src.ai.agent import Brain
from src.game.environment import Environment
from src.ai.utils.early_stop import EarlyStop
from src.config.settings import Config, config


class ReinforcementLearner:
    def __init__(
        self,
        config: Config,
        env: Environment,
        plotter: Optional[Plotter] = None,
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
        self.env: Environment = env
        self.gui: PygameGUI | None = gui
        self.plotter: Plotter | None = plotter

    # Training
    def train_epoch(self) -> Tuple[int, float, float, float]:
        """Runs a single training epoch."""
        best_score: int = 0
        best_time: float = 0
        total_rewards: float = 0
        total_loss: float = 0
        self.env.reset()
        start_time: float = time.time()

        while True:
            self.env.train_step()
            best_score = max(
                best_score, max(snake.size for snake in self.env.snakes)
            )
            total_rewards += max(snake.reward for snake in self.env.snakes)

            for snake in self.env.snakes:
                if snake.brain:
                    snake.brain.replay()

            if self.gui:
                self.gui.training_render(self.env.map)

            if all(not snake.alive for snake in self.env.snakes):
                for snake in self.env.snakes:
                    if snake.brain:
                        total_loss += snake.brain.loss_value
                        snake.brain.replay()
                        snake.brain.decay_epsilon()

                avg_loss = total_loss / len(self.env.snakes)
                elapsed_time: float = time.time() - start_time
                best_time = max(best_time, elapsed_time)
                break

            # time.sleep(0.1)

        return best_score, best_time, total_rewards, avg_loss

    def train(self) -> None:
        """Trains the agent over multiple epochs."""
        early_stop = EarlyStop(
            patience=config.nn.patience, min_delta=config.nn.min_delta
        )
        best_score: int = 0
        best_time: float = 0
        epsilon: float = self.config.nn.exploration.epsilon
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            TextColumn("• Epoch: {task.fields[epoch]}"),
            BarColumn(),
            TextColumn("• Best Score: {task.fields[best_score]}"),
            TextColumn("• Best Time: {task.fields[best_time]:.2f}s"),
            TextColumn("• Total Rewards: {task.fields[total_rewards]:.2f}"),
            TextColumn("• Avg Loss: {task.fields[avg_loss]:.2f}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                "Training",
                total=self.config.nn.epochs,
                epoch=0,
                best_score=0,
                best_time=0,
                total_rewards=0,
                avg_loss=0,
            )
            for epoch in range(self.config.nn.epochs):
                epoch_score, epoch_time, rewards, avg_loss = (
                    self.train_epoch()
                )
                best_score = max(best_score, epoch_score)
                best_time = max(best_time, epoch_time)
                progress.update(
                    task,
                    advance=1,
                    epoch=epoch + 1,
                    best_score=best_score,
                    best_time=best_time,
                    total_rewards=rewards,
                    avg_loss=avg_loss,
                )

                early_stop(avg_loss, self.env.snakes[0].brain)
                if early_stop.early_stop:
                    progress.update(
                        task,
                        description="Early Stopping Detected",
                        epoch=epoch + 1,
                        best_score=best_score,
                        best_time=best_time,
                        total_rewards=rewards,
                        avg_loss=avg_loss,
                    )
                    break

                for snake in self.env.snakes:
                    if snake.brain:
                        epsilon = min(epsilon, snake.brain.epsilon)

                if self.plotter:
                    self.plotter.update(
                        epoch, rewards, epoch_score, best_score, epsilon
                    )

                if epoch % config.nn.update_frequency == 0:
                    for snake in self.env.snakes:
                        if snake.brain:
                            snake.brain.save(
                                config.paths.models / "snake_brain.pth"
                            )

        if self.plotter:
            self.plotter.close()

    def print_epoch(
        self,
        epoch: int,
        best_score: int,
        best_time: float,
        rewards: float,
        avg_loss: float,
    ) -> None:
        """Prints the epoch results."""
        print(
            f"Epoch {epoch + 1}/{self.config.nn.epochs}, "
            f"Best Score: {best_score}, "
            f"Best Time: {best_time:.2f}s, "
            f"Total Rewards: {rewards}, "
            f"Avg Loss: {avg_loss:.2f}"
        )

    @torch.no_grad()
    def evaluate(self, test_episodes: int = 5) -> None:
        """Evaluates the agent's performance over multiple episodes."""
        for i in range(test_episodes):
            self.env.reset()
            max_score: int = 0

            while True:
                self.env.step()
                max_score = max(
                    max_score, max(snake.size for snake in self.env.snakes)
                )

                if self.gui:
                    self.gui.training_render(self.env.map)

                if all(not snake.alive for snake in self.env.snakes):
                    print(
                        f"Episode {i + 1}/{test_episodes}, "
                        f"Max Score: {max_score}, "
                    )
                    break

                # time.sleep(0.1)


if __name__ == "__main__":
    gui = PygameGUI(config)
    plotter = None
    # plotter = Plotter()
    model_path = config.paths.models / "snake_brain.pth"

    if not model_path.exists():
        model_path = None

    brain = Brain(config=config, path=model_path)
    snakes = Snake(1, brain=brain, config=config)
    env = Environment(config, [snakes])
    interpreter = ReinforcementLearner(
        config=config,
        gui=gui or None,
        env=env,
        plotter=plotter,
    )

    try:
        interpreter.train()
        interpreter.evaluate()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
