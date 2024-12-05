import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
import argparse
import pandas as pd
import os

from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from mygym.my_pendulum import PendulumRenderFix
from stable_baselines3.common.utils import get_linear_fn

from wrapper.calf_wrapper import CALFWrapper, CALFEnergyPendulumWrapper
from controller.energybased import EnergyBasedController
from stable_baselines3.common.vec_env import DummyVecEnv

os.makedirs("logs", exist_ok=True)


# Initialize the argument parser
parser = argparse.ArgumentParser(description="PPO Training and Evaluation for Pendulum")
parser.add_argument("--console", action="store_true", help="Disable graphical output for console-only mode")
parser.add_argument("--notrain", action="store_true", help="Skip the training phase")
parser.add_argument("--loadstep", 
                    type=int,
                    help="Choose step to load checkpoint")
parser.add_argument("--seed", 
                    type=int,
                    help="Choose random seed",
                    default=42)
# Parse the arguments
args = parser.parse_args()

matplotlib.use("TkAgg")  # Try "Qt5Agg" if "TkAgg" doesn't work

# Register the environment
gym.envs.registration.register(
    id="PendulumRenderFix-v0",
    entry_point="mygym.my_pendulum:PendulumRenderFix",
)

# Use your custom environment for training
env = gym.make("PendulumRenderFix-v0")

env = TimeLimit(env, max_episode_steps=1000)  # Set a maximum number of steps per episode

# Total number of agent-environment interaction steps for training
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
    "use_sde": True, # Whether to use generalized State Dependent Exploration (gSDE) instead of action noise exploration (default: False)
    "sde_sample_freq": 4, # Sample a new noise matrix every n steps when using gSDE
}

# More detailed explanation:
#
# learning_rate: Controls how quickly or slowly the model updates its parameters. A very low value, like 1e-6, results in slow learning, which can sometimes prevent instability.
# n_steps: Determines how many steps of experience are collected before updating the policy. A larger n_steps provides more data for each update but requires more memory and computation.
# batch_size: The number of samples used to compute each gradient update. It affects the variance of the gradient estimate and the stability of learning.
# gamma: The discount factor, which defines how future rewards are weighted relative to immediate rewards. A high value (close to 1) makes the agent focus on long-term rewards.
# gae_lambda: A parameter used in the Generalized Advantage Estimation (GAE) method, which helps reduce variance in the advantage estimates. It controls the trade-off between bias and variance.
# clip_range: The range within which the policy is clipped to prevent overly large updates, ensuring more stable training.

print("Skipping training phase...")

# ====Evaluation: animated plot to show trained agent's performance
def make_env(seed):
    def _init():
        env = PendulumRenderFix(render_mode="human" if not args.console else None)
        # env = PendulumRenderFix()
        # env = TimeLimit(env, max_episode_steps=1000)  # Set a maximum number of steps per episode
        env = CALFWrapper(
            env,
            fallback_policy=CALFEnergyPendulumWrapper(EnergyBasedController()),
            calf_decay_rate=0.01,
            initial_relax_prob=0.5,
            relax_prob_base_step_factor=0.95,
            relax_prob_episode_factor=0.,
            debug=False,
        )
        
        env.reset(seed=seed)
        
        return env
    return _init
# Now enable rendering with pygame for testing
import pygame
env_agent = DummyVecEnv([make_env(args.seed)])

# Load the model (if needed)
if not args.loadstep:
    model = PPO.load("ppo_pendulum")
else:
    model = PPO.load(f"checkpoints/ppo_pendulum_{args.loadstep}_steps")

# Reset the environment
env_agent.seed(seed=args.seed)
obs = env_agent.reset()
env_agent.env_method("copy_policy_model", model.policy)

# Initialize pygame and set the display size
pygame.init()
# screen = pygame.display.set_mode((800, 600))  # Adjust the dimensions as needed
info_dict = {
    "state": [],
    "action": [],
    "reward": [],
    "accumulated_reward": [],
    "relax_probability": [],
    "calf_activated_count": [],
}
accumulated_reward = 0

# Run the simulation and render it
for _ in range(1000):
    action, _ = model.predict(obs)
    # Dynamically handle four or five return values
    result = env_agent.step(action)  # Take a step in the environment
    if len(result) == 4:
        obs, reward, done, info = result
        truncated = False
    else:
        obs, reward, done, truncated, info = result

    accumulated_reward += reward

    info_dict["state"].append(obs)
    info_dict["action"].append(action)
    info_dict["reward"].append(reward)
    info_dict["accumulated_reward"].append(accumulated_reward.copy())
    info_dict["relax_probability"].append(env_agent.get_attr("relax_prob").copy()[0])
    info_dict["calf_activated_count"].append(env_agent.get_attr("calf_activated_count").copy()[0])

# Close the environment after the simulation
env_agent.close()

df = pd.DataFrame(info_dict)
checkpoint_name = args.loadstep if args.loadstep else "last_checkpoint"
file_name = f"pure_ppo_with_calfw_eval_{checkpoint_name}_seed_{args.seed}.csv"

if args.log:
    df.to_csv("logs/" + file_name)

print("Case:", file_name)
print(df.tail(2))
