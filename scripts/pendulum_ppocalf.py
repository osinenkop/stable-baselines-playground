import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
import argparse
import numpy as np
import time
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from gymnasium.wrappers import TimeLimit

# Import the custom callback from callback.py
from src.callback.plotting_callback import PlottingCallback
from src.controller.pid import PIDController
from src.controller.energybased import EnergyBasedController


# Initialize the argument parser
parser = argparse.ArgumentParser(description="PPO Training and Evaluation for Pendulum")
parser.add_argument("--notrain", action="store_true", help="Skip the training phase")

# Parse the arguments
args = parser.parse_args()

matplotlib.use("TkAgg")  # Try "Qt5Agg" if "TkAgg" doesn't work

# Register the environment
gym.envs.registration.register(
    id="PendulumRenderFix-v0",
    entry_point="src.mygym.my_pendulum:PendulumRenderFix",
)

# Use your custom environment for training
env = gym.make("PendulumRenderFix-v0")

env = TimeLimit(env, max_episode_steps=1000)  # Set a maximum number of steps per episode

# ---------------------------------
# Initialize the PID controller
kp = 5.0  # Proportional gain
ki = 0.1   # Integral gain
kd = 1.0   # Derivative gain
pid = PIDController(kp, ki, kd, setpoint=0.0)  # Setpoint is the upright position (angle = 0)

dt = 0.05  # Action time step for the simulation
# ---------------------------------
# Initialize the energy-based controller
controller = EnergyBasedController()
# ---------------------------------

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
}

# More detailed explanation:
#
# learning_rate: Controls how quickly or slowly the model updates its parameters. A very low value, like 1e-6, results in slow learning, which can sometimes prevent instability.
# n_steps: Determines how many steps of experience are collected before updating the policy. A larger n_steps provides more data for each update but requires more memory and computation.
# batch_size: The number of samples used to compute each gradient update. It affects the variance of the gradient estimate and the stability of learning.
# gamma: The discount factor, which defines how future rewards are weighted relative to immediate rewards. A high value (close to 1) makes the agent focus on long-term rewards.
# gae_lambda: A parameter used in the Generalized Advantage Estimation (GAE) method, which helps reduce variance in the advantage estimates. It controls the trade-off between bias and variance.
# clip_range: The range within which the policy is clipped to prevent overly large updates, ensuring more stable training.

# Check if the --notrain flag is provided
if not args.notrain:

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

    # Train the model
    print("Training the model...")
    model.learn(total_timesteps=total_timesteps, callback=plotting_callback)
    # Save the model after training
    model.save("artifacts/checkpoints/ppo_pendulum")
    # Close the plot after training
    plt.ioff()  # Turn off interactive mode
    # plt.show()  # Show the final plot
    # plt.close("all")   
else:
    print("Skipping training phase...")

# ====Evaluation: animated plot to show trained agent's performance

# Now enable rendering with pygame for testing
import pygame
env = gym.make("PendulumRenderFix-v0", render_mode="human")

# Load the model (if needed)
model = PPO.load("artifacts/checkpoints/ppo_pendulum")

# Reset the environment
# obs, _ = env.reset()
obs, _ = env.reset(options={"angle": np.pi, "angular_velocity": 1.0})
cos_angle, sin_angle, angular_velocity = obs

# Initialize pygame and set the display size
pygame.init()
# screen = pygame.display.set_mode((800, 600))  # Adjust the dimensions as needed

# Initial critic value (expand dims to simulate batch input)
obs_tensor = torch.tensor([obs], dtype=torch.float32)  # Add batch dimension and convert to tensor
previous_value = model.policy.predict_values(obs_tensor)

# Initialize total reward
total_reward = 0.0

# Run the simulation and render it
dt = 0.05  # Time step for the simulation
for step in range(500):
    # Compute the control action using the energy-based controller
    control_action = controller.compute(cos_angle, angular_velocity)
    control_action = np.clip([control_action], -2.0, 2.0)

    # Generate the action from the agent model
    agent_action, _ = model.predict(obs)
    agent_action = np.clip(agent_action, -2.0, 2.0)  # Clip to the valid range

    # Convert the current observation to a PyTorch tensor
    obs_tensor = torch.tensor([obs], dtype=torch.float32)

    # Evaluate the current state using the critic
    current_value = model.policy.predict_values(obs_tensor)

    # Compare the critic values to decide which action to use
    if current_value > previous_value:
        # Use the agent's action if the critic value has improved
        action = agent_action
    else:
        # Otherwise, fallback to the energy-based controller's action
        action = control_action

    # !DEBUG
    # action = control_action
    # !DEBUG

    # Update the previous value for the next iteration
    previous_value = current_value

    # Step the environment using the selected action
    obs, reward, done, _, _ = env.step(action)
    env.render()

    # Update the observation
    cos_angle, sin_angle, angular_velocity = obs

    # Update the total reward
    total_reward += reward

    # Formatted print statement
    print(f"Step: {step + 1:3d} | Current Reward: {reward:7.2f} | Total Reward: {total_reward:10.2f}")

    # Wait for the next time step
    time.sleep(dt)

# Close the environment after the simulation
env.close()
