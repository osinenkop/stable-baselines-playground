import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from ..mygym.my_pendulum import PendulumVisual

# Initialize the custom Pendulum environment
env = PendulumVisual()

# Reset the environment and get the initial image
image, _ = env.reset()

# Create a figure for displaying the image
plt.ion()  # Turn on interactive mode for live updates
fig, ax = plt.subplots()
img_plot = ax.imshow(image)

# Run the simulation with random actions
for _ in range(500):  # Run for 100 steps
    # Generate a random action within the valid range
    action = np.random.uniform(low=-2.0, high=2.0, size=(1,))
    
    # Take a step in the environment
    image, reward, done, truncated, _ = env.step(action)
    
    # Update the image plot
    img_plot.set_data(image)
    plt.pause(0.01)  # Pause briefly to update the plot

    # Check if the episode is done and reset if necessary
    if done or truncated:
        image, _ = env.reset()

# Close the environment
env.close()
plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the last frame displayed
