import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        # Get the number of input channels from the observation space
        n_input_channels = observation_space.shape[2]  # Should be 3 for RGB images

        # DEBUG
        print(f"n_input_channels = {n_input_channels}")

        # Define a simple CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )

        # Compute the output shape of the CNN
        with th.no_grad():
            # Create a sample input tensor with the correct shape
            sample_input = th.zeros(1, n_input_channels, 500, 500)  # Adjust to (1, channels, height, width)

            # DEBUG
            print(f"sample_input.shape = {sample_input.shape}")

            n_flatten = self.cnn(sample_input).shape[1]

        # Final fully connected layer
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))
