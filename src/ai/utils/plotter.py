"""Plotter module."""

import matplotlib.pyplot as plt
from typing import List

from src.config.settings import config


class Plotter:
    def __init__(self) -> None:
        """Initializes the Plotter for data tracking."""
        self.episodes: List[int] = []
        self.rewards: List[float] = []
        self.scores: List[int] = []
        self.top_score: List[int] = []
        self.epsilons: List[float] = []

        # Set up the interactive plot
        plt.ion()
        self.fig, self.axs = plt.subplots(1, 4, figsize=(20, 5))
        self.fig.suptitle("Training Progress", fontsize=16)

        # Subplot for rewards
        self.axs[0].set_title("Rewards Over Episodes")
        self.axs[0].set_xlabel("Episodes")
        self.axs[0].set_ylabel("Total Reward")
        self.axs[0].grid()

        # Subplot for scores
        self.axs[1].set_title("Scores Over Episodes")
        self.axs[1].set_xlabel("Episodes")
        self.axs[1].set_ylabel("Score")
        self.axs[1].grid()

        # Subplot for Top Score
        self.axs[3].set_title("Top Score Over Episodes")
        self.axs[3].set_xlabel("Episodes")
        self.axs[3].set_ylabel("Top Score")
        self.axs[3].grid()

        # Subplot for epsilon
        self.axs[2].set_title("Epsilon Decay")
        self.axs[2].set_xlabel("Episodes")
        self.axs[2].set_ylabel("Epsilon")
        self.axs[2].grid()

    def update(
        self,
        episode: int,
        reward: float,
        score: int,
        top_score: int,
        epsilon: float,
    ) -> None:
        """Updates the plot data and redraws the plot."""
        # Append new data
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.scores.append(score)
        self.top_score.append(top_score)
        self.epsilons.append(epsilon)

        # Update each subplot
        self.axs[0].cla()
        self.axs[0].plot(
            self.episodes, self.rewards, label="Total Reward", color="blue"
        )
        self.axs[0].set_title("Rewards Over Episodes")
        self.axs[0].set_xlabel("Episodes")
        self.axs[0].set_ylabel("Total Reward")
        self.axs[0].grid()

        # Clear the Scores subplot
        self.axs[1].cla()
        self.axs[1].plot(
            self.episodes, self.scores, label="Score", color="orange"
        )
        self.axs[1].set_title("Scores Over Episodes")
        self.axs[1].set_xlabel("Episodes")
        self.axs[1].set_ylabel("Score")
        self.axs[1].grid()

        # Clear the Top Score subplot
        self.axs[2].cla()
        self.axs[2].plot(
            self.episodes, self.top_score, label="Top Score", color="red"
        )
        self.axs[2].set_title("Top Score Over Episodes")
        self.axs[2].set_xlabel("Episodes")
        self.axs[2].set_ylabel("Top Score")
        self.axs[2].grid()

        # Clear the Epsilon subplot
        self.axs[3].cla()
        self.axs[3].plot(
            self.episodes, self.epsilons, label="Epsilon", color="green"
        )
        self.axs[3].set_title("Epsilon Decay")
        self.axs[3].set_xlabel("Episodes")
        self.axs[3].set_ylabel("Epsilon")
        self.axs[3].grid()

        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        """Closes the interactive plot window."""
        plt.ioff()
        plt.savefig(config.paths.outputs / "training_progress.png")
        plt.close()
