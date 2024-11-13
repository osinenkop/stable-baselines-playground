import torch as th
import numpy as np
import matplotlib.pyplot as plt
from model.cnn import CustomCNN
from mygym.my_pendulum import PendulumVisual  # Ensure the import path is correct

# Initialize the custom Pendulum environment
env = PendulumVisual()

# Reset the environment and get the initial image
image, _ = env.reset()

# Print the original image shape to confirm it
print("Original image shape:", image.shape)
print("Should be (500, 500, 3)")

# Convert the image to a PyTorch tensor, permute the dimensions, and add a batch dimension
image_tensor = th.tensor(image, dtype=th.float32).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, 500, 500)
print("Permuted image tensor shape:", image_tensor.shape)
print("Should be (1, 3, 500, 500)")

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
