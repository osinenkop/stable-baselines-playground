import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
import argparse
import numpy as np
import time
import pygame

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit
from mygym.my_pendulum import PendulumRenderFix
# Import the custom callback from callback.py
from callback.plotting_callback import PlottingCallback
from stable_baselines3.common.utils import get_linear_fn
from controller.pid import PIDController
from controller.energybased import EnergyBasedController

import pandas as pd


# Initialize the argument parser
parser = argparse.ArgumentParser(description="PPO Training and Evaluation for Pendulum")
parser.add_argument("--console", action="store_true", help="Disable graphical output for console-only mode")
parser.add_argument("--seed", 
                    type=int,
                    help="Choose random seed",
                    default=42)
# Parse the arguments
args = parser.parse_args()

print("Simulating controller on pendulum. PRESS SPACE TO PAUSE.")

if args.console:
    matplotlib.use('Agg')  # Use a non-GUI backend to disable graphical output
else:
    matplotlib.use("TkAgg")  # Try "Qt5Agg" if "TkAgg" doesn't work

# Register the environment
gym.envs.registration.register(
    id="PendulumRenderFix-v0",
    entry_point="mygym.my_pendulum:PendulumRenderFix",
)

env_display = gym.make("PendulumRenderFix-v0", render_mode="human" if not args.console else None)
# env_display = gym.make("PendulumRenderFix-v0")

np.random.seed(args.seed)
# env = gym.make("PendulumRenderFix-v0", render_mode="human")
high, low = env_display.observation_space.high, env_display.observation_space.low
options = {
    "angle": np.random.uniform(-np.pi, np.pi),
    "angular_velocity": np.random.uniform(low[-1], high[-1]),
}

# Reset the environment
obs, _ = env_display.reset(seed=args.seed, options=options)
cos_angle, sin_angle, angular_velocity = obs
angle = np.arctan2(sin_angle, cos_angle)

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

info_dict = {
    "state": [],
    "action": [],
    "reward": [],
    "accumulated_reward": [],
}
accumulated_reward = 0

# Initialize pygame and set the display size
pygame.init()
# screen = pygame.display.set_mode((800, 600))  # Adjust the dimensions as needed

paused = False  # Variable to track the pause state

# Run the simulation and render it
for _ in range(1000):

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            paused = not paused  # Toggle pause state
        elif event.type == pygame.QUIT:
            env_display.close()
            pygame.quit()  # Ensure Pygame quits properly
            exit()  # Exit cleanly on window close

    if paused:
        while paused:
            # Continue processing events while paused
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    paused = not paused  # Resume simulation
                elif event.type == pygame.QUIT:
                    env_display.close()
                    pygame.quit()
                    exit()
            env_display.render()  # Keep rendering during pause
            time.sleep(0.1)  # Add a small delay to prevent high CPU usage
        continue  # Resume simulation loop after unpausing

    # Compute the control action using the nominal controller
    control_action = controller.compute(angle, cos_angle, angular_velocity)

    # Clip the action to the valid range for the Pendulum environment
    action = np.clip([control_action], -2.0, 2.0)

    obs, reward, done, _, _ = env_display.step(action)

    # Update the observation
    cos_angle, sin_angle, angular_velocity = obs
    angle = np.arctan2(sin_angle, cos_angle)

    accumulated_reward += reward

    info_dict["state"].append(obs)
    info_dict["action"].append(action)
    info_dict["reward"].append(reward)
    info_dict["accumulated_reward"].append(accumulated_reward.copy())

    # # Wait for the next time step
    # time.sleep(dt)  

# Close the environment after the simulation
env_display.close()

df = pd.DataFrame(info_dict)
df.to_csv(f"logs/energy_based_run_seed_{args.seed}.csv")

print(df.tail(2))
