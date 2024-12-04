import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import time
import pygame
import os
import csv
from datetime import datetime
import argparse

from controller.energybased import EnergyBasedController

print("Simulating controller on pendulum. PRESS SPACE TO PAUSE.")

matplotlib.use("TkAgg")  # Use "TkAgg" or another backend compatible with your system.

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Simulate energy-based pendulum controller.")
parser.add_argument("--log", action="store_true", help="Enable logging and printing of simulation data.")
args = parser.parse_args()

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

dt = 0.05  # Action time step for the simulation

# ---------------------------------
# Initialize the energy-based controller
controller = EnergyBasedController()
# ---------------------------------

# Initialize logging if `--log` is enabled
if args.log:
    simdata_dir = "simdata"
    os.makedirs(simdata_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(simdata_dir, f"pendulum_energy_{timestamp}.csv")

    # Write CSV header
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "angle", "angular_velocity", "action", "reward", "accumulated_reward"])

# Initialize pygame and set the display size
pygame.init()

paused = False  # Variable to track the pause state

# Initialize reward tracking
accumulated_reward = 0.0

# Run the simulation and render it
for step in range(1500):

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
    action = np.clip(control_action, -2.0, 2.0)

    # Step the environment
    obs, reward, done, _, _ = env_display.step([action])

    # Convert reward to scalar (from numpy.ndarray)
    reward = float(reward)

    # Update the observation
    cos_angle, sin_angle, angular_velocity = obs
    angle = np.arctan2(sin_angle, cos_angle)

    # Accumulate rewards
    accumulated_reward += reward

    if args.log:
        # Log and print data
        t = step * dt
        log_entry = (
            f"| t: {t:.2f} | angle: {angle:.2f} | angular_velocity: {angular_velocity:.2f} | "
            f"action: {action:.2f} | reward: {reward:.2f} | accumulated_reward: {accumulated_reward:.2f} |"
        )
        print(log_entry)
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([t, angle, angular_velocity, action, reward, accumulated_reward])

    # Wait for the next time step
    time.sleep(dt)

# Close the environment after the simulation
env_display.close()
