import os
import torch
import matplotlib.pyplot as plt
import numpy as np
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

    def _save_frame_visualization(self, obs, features, step, reward, action, angular_velocity, time_step_ms):
        """
        Save a visualization combining raw observations and feature maps.

        Args:
            obs (np.ndarray): Stacked observations.
            features (dict): CNN layer outputs.
            step (int): Current training step.
            reward (float): Current reward.
            action (float): Current action.
            angular_velocity (float): True angular velocity.
            time_step_ms (float): Time step length in milliseconds.
        """
        # Extract raw RGB frames from stacked observations
        num_frames = obs.shape[0] // 3  # Assuming RGB channels
        for frame_idx in range(num_frames):
            frame = obs[3 * frame_idx: 3 * (frame_idx + 1)].transpose(1, 2, 0)  # HWC format for RGB
            
            # Create a figure with subplots
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Plot raw frame
            axes[0].imshow(frame.astype(np.uint8))
            axes[0].set_title(f"Frame {frame_idx + 1} (RGB)")
            axes[0].axis("off")
            
            # Plot CNN features
            for i, (layer_name, layer_features) in enumerate(features.items(), start=1):
                if i > 3:  # Plot only up to 3 layers
                    break
                feature_map = layer_features[0, 0].detach().cpu().numpy()  # First batch, first channel
                axes[i].imshow(feature_map, cmap="viridis")
                axes[i].set_title(f"{layer_name} (First Channel)")
                axes[i].axis("off")
            
            # Add a unified title with extra info
            fig.suptitle(
                f"Step {step}, Reward: {reward:.2f}, Action: {action:.2f}, "
                f"Angular Velocity: {angular_velocity:.2f}, Time Step: {time_step_ms:.2f}ms",
                fontsize=12,
            )
            
            # Save the figure
            save_path = os.path.join(self.visualization_dir, f"step_{step}_frame_{frame_idx + 1}.png")
            plt.savefig(save_path)
            plt.close(fig)

    def _on_step(self) -> bool:
        if self.n_calls % self.every_n_steps == 0:
            print("Saving CNN output...")

            # Fetch the most recent observation from the rollout buffer
            try:
                obs_sample = self.model.rollout_buffer.observations[-1]
                rewards_sample = self.model.rollout_buffer.rewards[-1]
                actions_sample = self.model.rollout_buffer.actions[-1]
            except AttributeError:
                print("Rollout buffer not found, skipping CNN saving.")
                return True

            # Get environment-specific info for angular velocity and time step
            env = self.model.get_env()
            angular_velocity = env.envs[0].state[1]  # Assuming the first environment provides state
            time_step_ms = env.envs[0].dt * 1000  # Convert time step to milliseconds

            # Ensure the observation is properly formatted
            obs_sample = torch.tensor(obs_sample, dtype=torch.float32).to(self.model.device)

            # Pass the observation through the CNN
            cnn_model = self.model.policy.features_extractor  # Custom CNN
            with torch.no_grad():
                layer_features = cnn_model.get_layer_features(obs_sample)

            # Save visualizations
            self._save_frame_visualization(
                obs_sample[0].cpu().numpy(), layer_features, self.num_timesteps,
                rewards_sample, actions_sample, angular_velocity, time_step_ms
            )

        return True
