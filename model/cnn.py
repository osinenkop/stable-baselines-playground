import torch
import torch.nn as nn
from gymnasium import spaces

class CustomCNN(nn.Module):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__()
        # Ensure that the observation space is an image with shape (C, H, W)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )


        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),  # Retain more spatial info
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Keep feature maps consistent
        #     nn.ReLU(),
        #     nn.Flatten()
        # )

        # Define your CNN layers
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten()
        # )

        # Define a simpler CNN with fewer layers
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=2, padding=1),  # Downsampling
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Further downsampling
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Final downsampling
        #     nn.ReLU(),
        #     nn.Flatten()
        # )        

        # Define a even simpler CNN
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=2, padding=1),  # Downsampling
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Further downsampling
        #     nn.ReLU(),
        #     nn.Flatten()  # Flatten for the fully connected layer
        # )


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
