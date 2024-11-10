import matplotlib
matplotlib.use("TkAgg")  # Try "Qt5Agg" if "TkAgg" doesn't work

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

from env.my_pendulum import PendulumRenderFix

# Register the environment
gym.envs.registration.register(
    id="PendulumRenderFix-v0",
    entry_point="env.my_pendulum:PendulumRenderFix",
)

# Define a custom callback for real-time plotting
class PlottingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PlottingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episodes = []
        plt.ion()  # Turn on interactive mode for live plotting
        self.figure, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], label="Episode Reward")
        self.ax.set_xlabel("Episodes")
        self.ax.set_ylabel("Rewards")
        self.ax.set_title("Live Learning Curve")
        self.ax.legend()
        plt.show(block=False)  # Show the plot window

    def _on_step(self) -> bool:
        # Check if an episode is done
        if "episode" in self.locals["infos"][0]:
            reward = self.locals["infos"][0]["episode"]["r"]
            self.episode_rewards.append(reward)
            self.episodes.append(len(self.episodes) + 1)

            # Update the plot
            self.line.set_xdata(self.episodes)
            self.line.set_ydata(self.episode_rewards)
            self.ax.relim()  # Recalculate limits
            self.ax.autoscale_view()  # Autoscale
            plt.draw()
            plt.pause(10.5)  # Pause to update the plot

        return True

# Use your custom environment for training
env = gym.make("PendulumRenderFix-v0")

# Define the hyperparameters for PPO
ppo_hyperparams = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
}

# Create the PPO model with the specified hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=ppo_hyperparams["learning_rate"],
    n_steps=ppo_hyperparams["n_steps"],
    batch_size=ppo_hyperparams["batch_size"],
    gamma=ppo_hyperparams["gamma"],
    gae_lambda=ppo_hyperparams["gae_lambda"],
    clip_range=ppo_hyperparams["clip_range"],
    verbose=1,
)

# Create the plotting callback
plotting_callback = PlottingCallback()

# Train the model with the callback
model.learn(total_timesteps=100000, callback=plotting_callback)

# Close the plot after training
plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot
