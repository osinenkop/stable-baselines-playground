import torch
import matplotlib.pyplot as plt
import os

# Specify the path to the saved outputs
output_path = "./cnn_outputs"

# Load and visualize CNN outputs
for file_name in sorted(os.listdir(output_path)):
    if file_name.endswith(".pt"):
        # Load both raw features and processed output
        processed_output, raw_features = torch.load(os.path.join(output_path, file_name))
        print(f"Loaded CNN features from {file_name}")
        
        # Visualize the raw features (example: up to 3 feature maps)
        for i in range(min(3, raw_features.shape[1])):
            plt.imshow(raw_features[0, i].cpu().numpy(), cmap="viridis")
            plt.title(f"{file_name}: Raw Feature Map {i}")
            plt.colorbar()
            plt.show()
