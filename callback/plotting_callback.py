# callback.py
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class PlottingCallback(BaseCallback):
    def __init__(self, verbose=0):

        # print("DEBUG: PLOT: INIT")

        self.rewards = []  # List to store rewards accumulated over steps
        self.steps = []  # List to store step numbers

        super(PlottingCallback, self).__init__(verbose)
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

        # print("DEBUG: PLOT: ON_STEP")

        # # Get the reward from the current step
        # reward = self.locals["rewards"]
        # self.rewards.append(reward)
        # self.steps.append(len(self.steps) + 1)

        # # Update the plot every step
        # self.line.set_xdata(self.steps)
        # self.line.set_ydata(self.rewards)
        # self.ax.relim()  # Recalculate limits
        # self.ax.autoscale_view()  # Autoscale the plot
        # plt.draw()
        # plt.pause(0.01)  # Pause to update the plot

        # Accumulate the reward from the current step
        reward = self.locals["rewards"]
        self.current_episode_reward += reward

        # Check if the episode has ended
        done = self.locals["dones"]
        info = self.locals["infos"][0]  # Assuming a single environment

        if done or info.get("TimeLimit.truncated", False):
            # Log the total reward for the episode
            self.episode_rewards.append(self.current_episode_reward)
            self.episodes.append(len(self.episodes) + 1)
            self.current_episode_reward = 0  # Reset for the next episode

            # Update the plot
            self.line.set_xdata(self.episodes)
            self.line.set_ydata(self.episode_rewards)
            self.ax.relim()  # Recalculate limits
            self.ax.autoscale_view()  # Autoscale the plot
            plt.draw()
            plt.pause(0.01)  # Pause to update the plot

        return True
