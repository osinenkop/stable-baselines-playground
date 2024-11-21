import torch
import matplotlib.pyplot as plt
import os

# Specify the path to the saved outputs
output_path = "./cnn_outputs"

# Load and visualize CNN layer features
for file_name in sorted(os.listdir(output_path)):
    if file_name.endswith(".pt"):
        layer_features = torch.load(os.path.join(output_path, file_name))
        print(f"Loaded CNN layer features from {file_name}")

        for layer_name, features in layer_features.items():
            print(f"{layer_name}: shape {features.shape}")
            
            # Ensure the shape is correct for visualization
            if len(features.shape) != 4:
                print(f"Skipping {layer_name}: Invalid shape {features.shape}")
                continue

            # Rearrange dimensions if needed (N, C, H, W)
            if features.size(1) == 32 and features.size(3) == 2:
                features = features.permute(0, 1, 3, 2)  # Swap spatial dimensions (H, W)

            # Visualize the first few channels of this layer
            for i in range(min(3, features.size(1))):  # Plot up to 3 channels
                feature_map = features[0, i].cpu().numpy()

                if feature_map.ndim == 2:  # Ensure it's a valid 2D map
                    plt.imshow(feature_map, cmap="viridis", aspect="auto")
                    plt.title(f"{file_name}: {layer_name} - Feature Map {i}")
                    plt.colorbar()
                    plt.show()
                else:
                    print(f"Skipping {layer_name}, channel {i}: Invalid feature map shape {feature_map.shape}")
