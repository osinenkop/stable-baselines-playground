import torch
import torch.nn as nn
from gymnasium import spaces

class CustomCNN(nn.Module):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__()
        # Ensure that the observation space is an image with shape (C, H, W)
        n_input_channels = observation_space.shape[0]

        # Define your CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the size of the flattened output from the CNN
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample_input).shape[1]

        # Define the linear layer to produce the desired feature dimension
        self.linear = nn.Linear(n_flatten, features_dim)

        # Set the features_dim attribute
        self.features_dim = features_dim        

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
