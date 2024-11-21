import os
import torch
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

class SaveCNNOutputCallback(BaseCallback):
    def __init__(self, save_path: str, every_n_steps=1000):
        super(SaveCNNOutputCallback, self).__init__()
        self.save_path = save_path
        self.every_n_steps = every_n_steps
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.every_n_steps == 0:
            print("Saving CNN output...")
            
            # Fetch the most recent observation from the rollout buffer
            try:
                obs_sample = self.model.rollout_buffer.observations[-1]
            except AttributeError:
                print("Rollout buffer not found, skipping CNN saving.")
                return True

            # Ensure the observation is properly formatted and permuted
            if isinstance(obs_sample, np.ndarray):  # If using NumPy arrays
                obs_sample = torch.tensor(obs_sample, dtype=torch.float32).to(self.model.device)
            elif isinstance(obs_sample, torch.Tensor):  # If using Torch tensors
                obs_sample = obs_sample.to(self.model.device)

            # Permute the observation tensor to match PyTorch's expected input shape (N, C, H, W)
            obs_sample = obs_sample.permute(0, 3, 1, 2)

            # Pass the observations through the CNN
            cnn_model = self.model.policy.features_extractor  # Access the custom CNN
            with torch.no_grad():
                layer_features = cnn_model.get_layer_features(obs_sample)

            # Debug prints for the observation
            print(f"Live observation for CNN: {obs_sample.shape} - Min: {obs_sample.min()} Max: {obs_sample.max()}")

            # Save the features
            torch.save(
                layer_features,
                os.path.join(self.save_path, f"cnn_layer_features_step_{self.num_timesteps}.pt")
            )
            print(f"Saved CNN outputs at timestep {self.num_timesteps}.")
        return True


    def _extract_cnn_features(self, observations):
        cnn_model = self.model.policy.features_extractor  # Access the custom CNN
        with torch.no_grad():
            # Permute the observation tensor to match PyTorch's format (N, C, H, W)
            obs_tensor = torch.tensor(observations, dtype=torch.float32, device=self.model.device).permute(0, 3, 1, 2)
            layer_features = cnn_model.get_layer_features(obs_tensor)
        return layer_features
