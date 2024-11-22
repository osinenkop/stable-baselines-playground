import os
import torch
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class SaveCNNOutputCallback(BaseCallback):
    def __init__(self, save_path: str, every_n_steps=1000, max_channels=3):
        """
        Save CNN layer outputs and their visualizations during training.

        Args:
            save_path (str): Directory to save the CNN features and visualizations.
            every_n_steps (int): Save frequency in terms of training steps.
            max_channels (int): Maximum number of channels to visualize per layer.
        """
        super(SaveCNNOutputCallback, self).__init__()
        self.save_path = save_path
        self.every_n_steps = every_n_steps
        self.max_channels = max_channels
        os.makedirs(save_path, exist_ok=True)

        # Directory for visualizations
        self.visualization_dir = os.path.join(save_path, "visualizations")
        os.makedirs(self.visualization_dir, exist_ok=True)

    def _save_visualizations(self, layer_features, step):
        """
        Save visualizations of CNN feature maps.

        Args:
            layer_features (dict): Dictionary of layer outputs from the CNN.
            step (int): Current training step for file naming.
        """
        for layer_name, features in layer_features.items():
            if len(features.shape) != 4:  # Ensure valid (N, C, H, W) shape
                print(f"Skipping {layer_name}: Invalid shape {features.shape}")
                continue

            # Process and save feature maps for the first few channels
            batch_size, num_channels, height, width = features.shape
            for i in range(min(self.max_channels, num_channels)):
                feature_map = features[0, i].cpu().numpy()  # Take the first sample's i-th channel

                plt.figure(figsize=(6, 6))
                plt.imshow(feature_map, cmap="viridis", aspect="auto")
                plt.title(f"Step {step}: {layer_name} - Feature Map {i}")
                plt.colorbar()

                # Save visualization
                save_path = os.path.join(
                    self.visualization_dir,
                    f"step_{step}_{layer_name}_channel_{i}.png"
                )
                plt.savefig(save_path)
                plt.close()
                print(f"Saved visualization: {save_path}")

    def _on_step(self) -> bool:
        if self.n_calls % self.every_n_steps == 0:
            print("Saving CNN output...")

            # Fetch the most recent observation from the rollout buffer
            try:
                obs_sample = self.model.rollout_buffer.observations[-1]
            except AttributeError:
                print("Rollout buffer not found, skipping CNN saving.")
                return True             

            # Ensure the observation is properly formatted
            if isinstance(obs_sample, np.ndarray):
                obs_sample = torch.tensor(obs_sample, dtype=torch.float32).to(self.model.device)
            elif isinstance(obs_sample, torch.Tensor):
                obs_sample = obs_sample.to(self.model.device)

            # Pass the observation through the CNN
            cnn_model = self.model.policy.features_extractor  # Custom CNN
            with torch.no_grad():
                layer_features = cnn_model.get_layer_features(obs_sample)

            # Save raw features
            raw_feature_path = os.path.join(self.save_path, f"cnn_layer_features_step_{self.num_timesteps}.pt")
            torch.save(layer_features, raw_feature_path)
            print(f"Saved CNN features: {raw_feature_path}")

            # Save visualizations
            self._save_visualizations(layer_features, self.num_timesteps)

        return True
