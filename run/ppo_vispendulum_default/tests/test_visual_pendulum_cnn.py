import torch as th
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces  # Import spaces from gymnasium
from model.cnn import CustomCNN
from mygym.my_pendulum import PendulumVisual 
from mygym.my_pendulum import ResizeObservation 

# Initialize the custom Pendulum environment
image_height=50
image_width=50
env = PendulumVisual()

# Wrap the environment to resize the observations
env = ResizeObservation(env, (image_height, image_width))

# Reset the environment and get the initial image
image, _ = env.reset()

# Print the original image shape to confirm it
print("Original image shape:", image.shape)
print(f"Should be ({image_height}, {image_width}, 3)")

# Convert the image to a PyTorch tensor, permute the dimensions to CHW, and add a batch dimension
image_tensor = th.tensor(image, dtype=th.float32).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, image_height, image_width)
print("Permuted image tensor shape:", image_tensor.shape)
print(f"Should be (1, 3, {image_height}, {image_width})")

# Update the observation space to match (3, image_height, image_width) for compatibility
env.observation_space = spaces.Box(low=0, high=255, shape=(3, image_height, image_width), dtype=np.uint8)

# Initialize the CNN
cnn = CustomCNN(env.observation_space)

# Run a forward pass through the CNN and print the output
cnn_output = cnn(image_tensor)
print("CNN Output Shape:", cnn_output.shape)
print("CNN Output:", cnn_output)

# Visualize the initial image to verify
plt.imshow(image)
plt.title("Initial Image from PendulumVisual Environment")
plt.show()

# Close the environment
env.close()
