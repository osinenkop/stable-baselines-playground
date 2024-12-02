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

print("Simulating controller on pendulum. PRESS SPACE TO PAUSE.")

matplotlib.use("TkAgg")  # Try "Qt5Agg" if "TkAgg" doesn't work

# Register the environment
gym.envs.registration.register(
    id="PendulumRenderFix-v0",
    entry_point="mygym.my_pendulum:PendulumRenderFix",
)

env_display = gym.make("PendulumRenderFix-v0", render_mode="human")

# Reset the environment
obs, _ = env_display.reset()
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

# Initialize pygame and set the display size
pygame.init()
# screen = pygame.display.set_mode((800, 600))  # Adjust the dimensions as needed

paused = False  # Variable to track the pause state

# Run the simulation and render it
for _ in range(1500):

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

     # Wait for the next time step
    time.sleep(dt)  

# Close the environment after the simulation
env_display.close()
