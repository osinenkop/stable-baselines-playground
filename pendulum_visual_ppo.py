import argparse
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from model.cnn import CustomCNN
from mygym.my_pendulum import PendulumVisual
from callback.plotting_callback import PlottingCallback  # Import your existing callback

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--notrain", action="store_true", help="Skip training and only run evaluation")
args = parser.parse_args()

# Initialize the custom Pendulum environment
env = PendulumVisual()

# Initialize and wrap the custom Pendulum environment
env = DummyVecEnv([lambda: PendulumVisual()])  # Vectorize the environment
env = VecTransposeImage(env)  # Transpose image observations for PyTorch

# DEBUG
obs = env.reset()
print(f"obs.shape = {obs.shape}")

# Set random seed for reproducibility
set_random_seed(42)
env.seed(42)  # Use the updated seed method

# Define the policy_kwargs to use the custom CNN
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256)
)

# Create the PPO agent using the custom feature extractor
model = PPO(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2
)

# Total number of agent-environment interaction steps for training
total_timesteps = 500000

# Train the model if --notrain flag is not provided
if not args.notrain:
    plotting_callback = PlottingCallback()
    model.learn(total_timesteps=total_timesteps, callback=plotting_callback)
    model.save("ppo_visual_pendulum")
else:
    print("Skipping training. Loading the saved model...")
    model = PPO.load("ppo_visual_pendulum")

# Close the plot
plt.close()

# Visual evaluation after training or loading
print("Starting visual evaluation...")

# Visual evaluation after training or loading
env = DummyVecEnv([lambda: PendulumVisual()])
env = VecTransposeImage(env)
obs = env.reset()

# Reload the environment for evaluation
env = PendulumVisual()
obs, _ = env.reset()

# Run the simulation with the trained agent
for _ in range(500):
    action, _ = model.predict(obs)
    obs, _, done, _, _ = env.step(action)
    env.render()
    if done:
        obs, _ = env.reset()

# Close the environment
env.close()
