import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Tuple

class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for stacked frame input.
    """

    def __init__(self, observation_space, features_dim: int = 256, num_frames: int = 4):
        # Get the shape of the input from the observation space
        input_channels = num_frames * 3  # Assuming RGB images
        super(CustomCNN, self).__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the size of the output from the CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, observation_space.shape[1], observation_space.shape[2])
            n_flatten = self.cnn(dummy_input).shape[1]

        # Linear layer to map the CNN output to the desired feature size
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

    def get_layer_features(self, x):
        """
        Extract intermediate feature maps from the CNN.

        Args:
            x (torch.Tensor): Input tensor to the CNN.
        
        Returns:
            dict: A dictionary containing feature maps at different layers.
        """
        features = {}
        
        # Pass through the first Conv2d layer
        x = self.cnn[0](x)  # Conv2d(32, ...)
        features["layer1"] = x.clone()  # Save feature map
        print(f"Layer1 output shape: {x.shape}")  # Optional debug print
        
        # Apply ReLU activation
        x = self.cnn[1](x)  # ReLU
        
        # Pass through the second Conv2d layer
        x = self.cnn[2](x)  # Conv2d(64, ...)
        features["layer2"] = x.clone()  # Save feature map
        print(f"Layer2 output shape: {x.shape}")  # Optional debug print
        
        # Apply ReLU activation
        x = self.cnn[3](x)  # ReLU
        
        # Pass through the third Conv2d layer
        x = self.cnn[4](x)  # Conv2d(128, ...)
        features["layer3"] = x.clone()  # Save feature map
        print(f"Layer3 output shape: {x.shape}")  # Optional debug print
        
        # Apply ReLU activation
        x = self.cnn[5](x)  # ReLU

        # Return all saved feature maps
        return features

# class CustomCNN(nn.Module):
#     def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
#         super(CustomCNN, self).__init__()
#         # Libraries like Stable-Baselines3 or environments often use the HWC format
#         # (Height, Width, Channels), whereas PyTorch's CNN layers expect the CHW format (Channels, Height, Width).
#         n_input_channels = observation_space.shape[0]

#         # print(f"n_input_channels = {n_input_channels}")

#         # print(f"observation_space.shape = {observation_space.shape}")

#         # Define the convolutional layers without the Flatten operation
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#         )

#         # Compute the size of the flattened output
#         with torch.no_grad():
#             # Libraries like Stable-Baselines3 or environments often use the HWC format
#             # (Height, Width, Channels), whereas PyTorch's CNN layers expect the CHW format (Channels, Height, Width). 
#             # Here, we fix it          

#             # print(f"Before torch.permute: sample_input.shape = {observation_space.shape}")

#             # sample_input = torch.zeros(1, *observation_space.shape).permute(0, 3, 1, 2)
#             sample_input = torch.zeros(1, *observation_space.shape)           

#             # print(f"After torch.permute: sample_input.shape = {sample_input.shape}")

#             n_flatten = self.cnn(sample_input).reshape(sample_input.size(0), -1).shape[1]

#         # Define the Flatten operation and the linear layer separately
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear(n_flatten, features_dim)

#         # Set the features_dim attribute
#         self.features_dim = features_dim

#     def forward(self, observation: torch.Tensor) -> torch.Tensor:
#         # Debugging: Print observation shape before correction
#         # print(f"Before permute: observation.shape={observation.shape}")
        
#         if observation.shape[1] == 3:  # Channels are already the second dimension
#             corrected_observation = observation  # No permutation needed
#         else:
#             corrected_observation = observation.permute(0, 3, 1, 2)  # Convert HWC to CHW

#         # Debugging: Print corrected observation
#         # print(f"Corrected observation shape: {corrected_observation.shape}")
#         # print(f"Corrected observation stats - Min: {corrected_observation.min()}, Max: {corrected_observation.max()}")

#         # Debug: plot image as perceived by the CNN
#         # image = observation[-1, :, :, :].permute(1, 2, 0)
#         # plt.imshow(image)
#         # plt.title("Observation from PendulumVisual as perceived by CNN")
#         # plt.axis('off')  # Hide axes
#         # plt.show()  # Block execution until the plot is closed  

#         # Debug prints for the observation
#         # print(f"Live observation for CNN (inside forward): {corrected_observation.shape} - Min: {corrected_observation.min()} Max: {corrected_observation.max()}")        

#         raw_features = self.cnn(corrected_observation)  # Preserve the spatial dimensions here
#         flattened_features = self.flatten(raw_features)  # Flatten for the linear layer
#         return self.linear(flattened_features)

#     def get_layer_features(self, x):
#         # print(f"Input shape: {x.shape}")  # Debugging print
#         features = {}
#         x = self.cnn[0](x)  # Conv2d(32, ...)
#         features["layer1"] = x.clone()
#         # print(f"Layer1 output shape: {x.shape}")  # Debugging print
#         x = self.cnn[1](x)  # ReLU
#         x = self.cnn[2](x)  # Conv2d(64, ...)
#         features["layer2"] = x.clone()
#         # print(f"Layer2 output shape: {x.shape}")  # Debugging print
#         x = self.cnn[3](x)  # ReLU
#         x = self.cnn[4](x)  # Conv2d(128, ...)
#         features["layer3"] = x.clone()
#         # print(f"Layer3 output shape: {x.shape}")  # Debugging print
#         x = self.cnn[5](x)  # ReLU
#         return features
