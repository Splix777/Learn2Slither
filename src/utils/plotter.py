import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        """
        Initializes the Plotter with empty lists for data tracking.
        """
        self.episodes = []
        self.rewards = []
        self.scores = []
        self.epsilons = []

    def update(self, episode, reward, score, epsilon):
        """
        Updates the plot data for the current episode.
        Args:
            episode (int): Current episode number.
            reward (float): Total reward for the episode.
            score (int): Final score for the episode.
            epsilon (float): Current epsilon value.
        """
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.scores.append(score)
        self.epsilons.append(epsilon)
        self.plot()

    def plot(self):
        """
        Plots the current progress.
        """
        plt.figure(figsize=(12, 6))
        plt.clf()

        # Plot Total Rewards
        plt.subplot(1, 3, 1)
        plt.plot(self.episodes, self.rewards, label="Total Reward")
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title("Rewards Over Episodes")
        plt.grid()

        # Plot Scores
        plt.subplot(1, 3, 2)
        plt.plot(self.episodes, self.scores, label="Score", color="orange")
        plt.xlabel("Episodes")
        plt.ylabel("Score")
        plt.title("Scores Over Episodes")
        plt.grid()

        # Plot Epsilon
        plt.subplot(1, 3, 3)
        plt.plot(self.episodes, self.epsilons, label="Epsilon", color="green")
        plt.xlabel("Episodes")
        plt.ylabel("Epsilon")
        plt.title("Epsilon Decay")
        plt.grid()

        plt.tight_layout()
        plt.pause(0.1)