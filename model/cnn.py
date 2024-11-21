import torch
import torch.nn as nn
from gymnasium import spaces

class CustomCNN(nn.Module):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__()
        # Ensure that the observation space is an image with shape (C, H, W)
        n_input_channels = observation_space.shape[0]

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
            sample_input = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample_input).view(sample_input.size(0), -1).shape[1]

        # Define the Flatten operation and the linear layer separately
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(n_flatten, features_dim)

        # Set the features_dim attribute
        self.features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Pass observations through the CNN and linear layers
        raw_features = self.cnn(observations)  # Preserve the spatial dimensions here
        flattened_features = self.flatten(raw_features)  # Flatten for the linear layer
        return self.linear(flattened_features)

    def get_intermediate_features(self, x):
        # Extract raw features (before flattening) and processed output
        raw_features = self.cnn(x)
        print(f"Raw features shape: {raw_features.shape}")  # Should now retain spatial dimensions
        processed_output = self.linear(self.flatten(raw_features))
        return raw_features, processed_output

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
