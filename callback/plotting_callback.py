# callback.py
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

class PlottingCallback(BaseCallback):
    def __init__(self, update_every_episodes=1, verbose=0):

        # print("DEBUG: PLOT: INIT")

        self.rewards = []  # List to store rewards accumulated over steps
        self.steps = []  # List to store step numbers

        super(PlottingCallback, self).__init__(verbose)
        self.update_every_episodes = update_every_episodes
        self.episode_rewards = []  # List to store rewards for each episode
        self.episodes = []  # List to store episode numbers
        self.current_episode_reward = 0  # Accumulator for the current episode reward

        # Set up the plot
        self.figure, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], label="Episode Reward")
        self.ax.set_xlabel("Episodes")
        self.ax.set_ylabel("Rewards")
        self.ax.set_title("Live Learning Curve")
        self.ax.legend()
        plt.show(block=False)  # Show the plot window without blocking

    def _on_step(self) -> bool:
        # Accumulate the reward from the current step
        rewards = self.locals["rewards"]
        self.current_episode_reward += np.sum(rewards)  # Sum rewards from all environments

        # Check if any episode has ended
        dones = self.locals["dones"]
        infos = self.locals["infos"]  # This is now a list of dictionaries

        if np.any(dones):
            # Check if any episode was truncated or ended due to a time limit
            if np.any([info.get("TimeLimit.truncated", False) for info in infos]):
                # Log the total reward for the episode
                self.episode_rewards.append(self.current_episode_reward)
                self.episodes.append(len(self.episodes) + 1)
                self.current_episode_reward = 0  # Reset for the next episode

                # Only update the plot every `update_every_episodes` episodes
                if len(self.episodes) % self.update_every_episodes == 0:
                    self.line.set_xdata(self.episodes)
                    self.line.set_ydata(self.episode_rewards)
                    self.ax.relim()
                    self.ax.autoscale_view()
                    plt.draw()
                    plt.pause(0.01)

        return True

