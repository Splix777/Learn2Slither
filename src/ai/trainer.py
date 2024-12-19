"""Agent Interface for the Snake Game."""

import os
import time
from typing import Optional, Tuple

import torch
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TaskID,
)
from rich.prompt import Prompt

from src.ui.pygame.pygame_gui import PygameGUI
from src.ai.utils.plotter import Plotter
from src.game.snake import Snake
from src.game.environment import Environment
from src.ai.utils.early_stop import EarlyStop
from src.config.settings import Config


class ReinforcementLearner:
    def __init__(
        self,
        config: Config,
        env: Environment,
        early_stop: Optional[bool] = None,
        plotter: Optional[Plotter] = None,
        gui: Optional[PygameGUI] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Initializes the AI agent for Snake."""
        self.config: Config = config
        self.env: Environment = env
        self.early_stop: bool | None = early_stop
        self.gui: PygameGUI | None = gui
        self.plotter: Plotter | None = plotter
        self.save_path: str | None = save_path

    # Training
    def train_epoch(self, fast: bool) -> Tuple:
        """Runs a single training epoch."""
        tscore: int = 0
        best_time: float = 0
        total_rewards: float = 0
        total_loss: float = 0
        self.env.reset()
        start_time: float = time.time()
        epsilon: float = self.config.nn.exploration.epsilon

        while True:
            self.env.train_step()
            tscore = max(tscore, max(snake.size for snake in self.env.snakes))
            total_rewards += max(snake.reward for snake in self.env.snakes)

            for snake in self.env.snakes:
                if snake.brain:
                    snake.brain.replay()

            if self.gui:
                self.gui.render_map(self.env)

            if all(not snake.alive for snake in self.env.snakes):
                for snake in self.env.snakes:
                    if snake.brain:
                        total_loss += snake.brain.loss_value
                        snake.brain.decay_epsilon()
                        epsilon = snake.brain.epsilon

                avg_loss = total_loss / len(self.env.snakes)
                elapsed_time: float = time.time() - start_time
                best_time = max(best_time, elapsed_time)
                break

            if not fast:
                time.sleep(0.1)

        return tscore, best_time, total_rewards, avg_loss, epsilon

    def train(self, fast: bool = True) -> None:
        """Trains the agent over multiple epochs."""
        early_stop = EarlyStop(
            patience=self.config.nn.patience,
            min_delta=self.config.nn.min_delta,
        )
        best_score: int = 0
        best_time: float = 0
        with self._set_train_pb() as progress:
            task: TaskID = self._train_task(progress)
            for epoch in range(self.config.nn.epochs):
                score, etime, reward, loss, epsilon = self.train_epoch(fast)
                best_score = max(best_score, score)
                best_time = max(best_time, etime)
                progress.update(
                    task,
                    advance=1,
                    epoch=epoch + 1,
                    best_score=best_score,
                    best_time=best_time,
                    total_rewards=reward,
                    avg_loss=loss,
                    epsilon=epsilon,
                )

                early_stop(loss, self.env.snakes[0].brain)
                if early_stop.early_stop and self.early_stop:
                    progress.update(
                        task,
                        description="Early Stopping Detected",
                        epoch=epoch + 1,
                        best_score=best_score,
                        best_time=best_time,
                        total_rewards=reward,
                        avg_loss=loss,
                        epsilon=epsilon,
                    )
                    for snake in self.env.snakes:
                        self._save_model(snake, "snake_brain.pth")
                    break

                for snake in self.env.snakes:
                    if snake.brain:
                        epsilon = min(epsilon, snake.brain.epsilon)
                if self.plotter:
                    self.plotter.update(
                        epoch, reward, score, best_score, epsilon
                    )
                if epoch in [10, 50, 100]:
                    for snake in self.env.snakes:
                        self._save_model(snake, f"snake_brain_{epoch}.pth")

        if self.plotter:
            self.plotter.close()
        if self.save_path:
            for snake in self.env.snakes:
                self._save_model(snake, self.save_path)

    def _save_model(self, snake: Snake, file_name: str) -> None:
        """Saves the model at a given epoch."""
        if snake.brain:
            snake.brain.save(self.config.paths.models / file_name)

    @torch.no_grad()
    def evaluate(
        self,
        test_episodes: int = 5,
        fast: bool = False,
        step_by_step: bool = False,
    ) -> None:
        """Evaluates the agent's performance over multiple episodes."""
        avg_score: float = 0
        best_run: int = 0
        max_score: int = 0

        with self._set_eval_pb() as progress:
            task: TaskID = self._eval_task(progress, test_episodes)
            for i in range(test_episodes):
                self.env.reset()

                while True:
                    self.env.step()
                    max_score = max(
                        max_score,
                        max(snake.size for snake in self.env.snakes),
                    )

                    if self.gui:
                        self.gui.render_map(self.env)

                    if step_by_step:
                        step = Prompt.ask("", default="")
                        os.system("clear" if os.name == "posix" else "cls")
                        if step.lower() in ["q", "quit", "exit"]:
                            return

                    if all(not snake.alive for snake in self.env.snakes):
                        avg_score += max_score
                        best_run = max(best_run, max_score)
                        current_avg = avg_score / (i + 1)
                        progress.update(
                            task,
                            advance=1,
                            episode=i + 1,
                            best_run=best_run,
                            avg_score=current_avg,
                        )
                        break

                    if not fast:
                        time.sleep(0.1)

    def _set_eval_pb(self) -> Progress:
        """Sets the progress bar for evaluation."""
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            TextColumn("• Episode: {task.fields[episode]}"),
            TextColumn("• Best Run: {task.fields[best_run]}"),
            TextColumn("• Avg Score: {task.fields[avg_score]:.2f}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )

    def _eval_task(self, progress: Progress, total: int) -> TaskID:
        """Creates a task for the progress bar."""
        return progress.add_task(
            "Evaluation", total=total, episode=0, best_run=0, avg_score=0
        )

    def _set_train_pb(self) -> Progress:
        """Sets the progress bar for training."""
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            TextColumn("• Epoch: {task.fields[epoch]}"),
            BarColumn(),
            TextColumn("• Best Score: {task.fields[best_score]}"),
            TextColumn("• Best Time: {task.fields[best_time]:.2f}s"),
            TextColumn("• Total Rewards: {task.fields[total_rewards]:.2f}"),
            TextColumn("• Avg Loss: {task.fields[avg_loss]:.2f}"),
            TextColumn("• Current Epsilon: {task.fields[epsilon]:.2f}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )

    def _train_task(self, progress: Progress) -> TaskID:
        """Creates a task for the progress bar."""
        return progress.add_task(
            "Training",
            total=self.config.nn.epochs,
            epoch=0,
            best_score=0,
            best_time=0,
            total_rewards=0,
            avg_loss=0,
            epsilon=self.config.nn.exploration.epsilon,
        )
