import gymnasium as gym
import argparse

from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3 import PPO

import torch

class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


# Initialize the argument parser
parser = argparse.ArgumentParser(description="PPO Training and Evaluation for Pendulum with image observation")
parser.add_argument("--notrain", action="store_true", help="Skip the training phase")

# Parse the arguments
args = parser.parse_args()

total_timesteps = 500000

# Define the hyperparameters for PPO
ppo_hyperparams = {
    "learning_rate": 5e-4,  # The step size used to update the policy network. Lower values can make learning more stable.
    "n_steps": 4000,  # Number of steps to collect before performing a policy update. Larger values may lead to more stable updates.
    "batch_size": 200,  # Number of samples used in each update. Smaller values can lead to higher variance, while larger values stabilize learning.
    "gamma": 0.98,  # Discount factor for future rewards. Closer to 1 means the agent places more emphasis on long-term rewards.
    "gae_lambda": 0.9,  # Generalized Advantage Estimation (GAE) parameter. Balances bias vs. variance; lower values favor bias.
    "clip_range": 0.05,  # Clipping range for the PPO objective to prevent large policy updates. Keeps updates more conservative.
    "learning_rate": get_linear_fn(5e-4, 1e-6, total_timesteps*2),  # Linear decay from 5e-5 to 1e-6
    "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
}

# More detailed explanation:
#
# learning_rate: Controls how quickly or slowly the model updates its parameters. A very low value, like 1e-6, results in slow learning, which can sometimes prevent instability.
# n_steps: Determines how many steps of experience are collected before updating the policy. A larger n_steps provides more data for each update but requires more memory and computation.
# batch_size: The number of samples used to compute each gradient update. It affects the variance of the gradient estimate and the stability of learning.
# gamma: The discount factor, which defines how future rewards are weighted relative to immediate rewards. A high value (close to 1) makes the agent focus on long-term rewards.
# gae_lambda: A parameter used in the Generalized Advantage Estimation (GAE) method, which helps reduce variance in the advantage estimates. It controls the trade-off between bias and variance.
# clip_range: The range within which the policy is clipped to prevent overly large updates, ensuring more stable training.


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use your gym env for Pendulum
env_name='Pendulum-v1'
env=gym.make(env_name, render_mode="rgb_array", g=9.81)

seeds = ppo_hyperparams['seeds']

# Check if the --notrain flag is provided
if not args.notrain:
    # We will train for different seeds
    for seed in seeds:
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
            tensorboard_log="./log/",
            seed=seed
        )

        model.calf_filter.reset()
        model.policy.to(device)
        model.calf_filter.init_policy(model.policy)

        # Train the model
        model_name = "Pedulum_PPO_" + str(seed)
        print("Training model " + model_name)
        model.learn(total_timesteps=total_timesteps, tb_log_name=model_name)

        # Save the model after training
        model.save("./SavedModels/" + model_name)
    
        # Close the training environment
        env.close()
else:
    print("Skipping training phase...")