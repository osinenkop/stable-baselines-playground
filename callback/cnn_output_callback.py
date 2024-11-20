import os
import torch
from stable_baselines3.common.callbacks import BaseCallback

class SaveCNNOutputCallback(BaseCallback):
    def __init__(self, save_path: str, obs_sample=None, every_n_steps=1000):
        super(SaveCNNOutputCallback, self).__init__()
        self.save_path = save_path
        self.obs_sample = obs_sample
        self.every_n_steps = every_n_steps
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Save the CNN outputs every `self.every_n_steps` steps
        if self.n_calls % self.every_n_steps == 0:
            if self.obs_sample is not None:
                # Pass the observations through the CNN
                cnn_features = self._extract_cnn_features(self.obs_sample)
                # Save the features to a file
                torch.save(
                    cnn_features,
                    os.path.join(self.save_path, f"cnn_output_step_{self.num_timesteps}.pt")
                )
                self.logger.info(f"Saved CNN outputs at timestep {self.num_timesteps}.")
        return True

    def _extract_cnn_features(self, observations):
        # Access the policy's CNN
        cnn_model = self.model.policy.features_extractor
        with torch.no_grad():
            # Convert the observations to a tensor
            obs_tensor = torch.tensor(observations, dtype=torch.float32, device=self.model.device)
            # Pass the observations through the CNN
            cnn_features = cnn_model(obs_tensor)
        return cnn_features
