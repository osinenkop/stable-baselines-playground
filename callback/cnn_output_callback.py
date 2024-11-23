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
        Save visualizations of the input frames and corresponding CNN features.
        """
        save_path = os.path.join(self.save_path, f"frame_step_{step}.png")
        num_channels = obs.shape[0]  # Total number of input channels (stacked frames)

        fig, axs = plt.subplots(2, num_channels, figsize=(4 * num_channels, 8))

        for idx in range(num_channels):
            # Handle frame dimensions dynamically
            input_frame = obs[idx]  # Slice the idx-th channel
            input_frame = np.squeeze(input_frame)  # Ensure it's 2D (H, W) for visualization

            # Plot input frame
            axs[0, idx].imshow(input_frame, cmap="viridis")
            axs[0, idx].set_title(f"Input Frame {idx + 1}")
            axs[0, idx].axis("off")

            # Plot CNN feature map for the respective channel (if available)
            if idx < features["layer1"].shape[1]:  # Prevent indexing out of bounds
                feature_map = features["layer1"][0, idx].detach().cpu().numpy()
                axs[1, idx].imshow(feature_map, cmap="viridis")
                axs[1, idx].set_title(f"Feature Map {idx + 1}")
                axs[1, idx].axis("off")

        fig.suptitle(
            f"Step {step}, Reward: {float(reward):.2f}, Action: {float(action):.2f}, "
            f"Angular Velocity: {angular_velocity:.2f}, Time Step: {time_step_ms:.2f} ms",
            fontsize=16
        )
        plt.tight_layout()
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
