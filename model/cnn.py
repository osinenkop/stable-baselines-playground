import torch
import torch.nn as nn
import sys

from gymnasium import spaces

class CustomCNN(nn.Module):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__()
        # Libraries like Stable-Baselines3 or environments often use the HWC format
        # (Height, Width, Channels), whereas PyTorch's CNN layers expect the CHW format (Channels, Height, Width).
        n_input_channels = observation_space.shape[2]

        # print(f"n_input_channels = {n_input_channels}")

        # print(f"observation_space.shape = {observation_space.shape}")

        # Define the convolutional layers without the Flatten operation
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Compute the size of the flattened output
        with torch.no_grad():
            # Libraries like Stable-Baselines3 or environments often use the HWC format
            # (Height, Width, Channels), whereas PyTorch's CNN layers expect the CHW format (Channels, Height, Width). 
            # Here, we fix it          
            sample_input = torch.zeros(1, *observation_space.shape).permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample_input).reshape(sample_input.size(0), -1).shape[1]

        # Define the Flatten operation and the linear layer separately
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(n_flatten, features_dim)

        # Set the features_dim attribute
        self.features_dim = features_dim

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Pass observations through the CNN and linear layers
        corrected_observation = observation.permute(0, 3, 1, 2)  # Swap axes to (batch_size, channels, height, width)

        # print(f"observation.shape = {observation.shape}")
        # print(f"corrected_observation.shape = {corrected_observation.shape}")

        raw_features = self.cnn(corrected_observation)  # Preserve the spatial dimensions here
        flattened_features = self.flatten(raw_features)  # Flatten for the linear layer
        return self.linear(flattened_features)

    def get_layer_features(self, x):
        print(f"Input shape: {x.shape}")  # Debugging print
        features = {}
        x = self.cnn[0](x)  # Conv2d(32, ...)
        features["layer1"] = x.clone()
        print(f"Layer1 output shape: {x.shape}")  # Debugging print
        x = self.cnn[1](x)  # ReLU
        x = self.cnn[2](x)  # Conv2d(64, ...)
        features["layer2"] = x.clone()
        print(f"Layer2 output shape: {x.shape}")  # Debugging print
        x = self.cnn[3](x)  # ReLU
        x = self.cnn[4](x)  # Conv2d(128, ...)
        features["layer3"] = x.clone()
        print(f"Layer3 output shape: {x.shape}")  # Debugging print
        x = self.cnn[5](x)  # ReLU
        return features

# class SimplifiedCNN(nn.Module):
#     def __init__(self, observation_space, features_dim=256):
#         super(SimplifiedCNN, self).__init__()
#         # Get the shape of the input (C, H, W)
#         input_shape = observation_space.shape

#         # Define a simpler CNN with fewer layers
#         self.cnn = nn.Sequential(
#             nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=2, padding=1),  # Downsampling
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Further downsampling
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Final downsampling
#             nn.ReLU(),
#             nn.Flatten()
#         )

#         # Calculate the output size of the CNN
#         with torch.no_grad():
#             sample_input = torch.zeros(1, *input_shape)
#             cnn_output_size = self.cnn(sample_input).shape[1]

#         # Linear layer to map the CNN output to the desired feature dimension
#         self.linear = nn.Linear(cnn_output_size, features_dim)

#     def forward(self, observations):
#         x = self.cnn(observations)
#         return self.linear(x)
