import torch
import matplotlib.pyplot as plt
import os

# Specify the path to the saved outputs
output_path = "./cnn_outputs"

# Load and visualize CNN outputs
for file_name in sorted(os.listdir(output_path)):
    if file_name.endswith(".pt"):
        cnn_features = torch.load(os.path.join(output_path, file_name))
        print(f"Loaded CNN features from {file_name} with shape {cnn_features.shape}")
        
        # Example: Visualize the features of the first channel
        for i in range(min(3, cnn_features.shape[1])):  # Visualize up to 3 feature maps
            plt.imshow(cnn_features[0, i].cpu().numpy(), cmap="viridis")
            plt.title(f"{file_name}: Feature Map {i}")
            plt.colorbar()
            plt.show()
