import torch
import torch.nn as nn
import gym

class CustomCNN(nn.Module):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__()
        # Ensure the observation space is an image with shape (H, W, C)
        assert len(observation_space.shape) == 3, "Observation space must be an image (H, W, C)"
        
        # Correctly extract the number of channels
        n_input_channels = observation_space.shape[2]  # Use shape[2] for the number of channels
        print("n_input_channels =", n_input_channels)  # Debug print to verify

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the shape by doing a forward pass with a dummy tensor
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, n_input_channels, 500, 500)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        self.features_dim = features_dim

    def forward(self, observations):
        return self.linear(self.cnn(observations))
