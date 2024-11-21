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
        print(f"Step: {self.n_calls}, Save Every: {self.every_n_steps}")
        if self.n_calls % self.every_n_steps == 0:
            print("Saving CNN output...")
            if self.obs_sample is not None:
                # Pass the observations through the CNN
                processed_output, raw_features = self._extract_cnn_features(self.obs_sample)
                # Save the features as a tuple
                torch.save(
                    (processed_output, raw_features),  # Save both processed output and raw features
                    os.path.join(self.save_path, f"cnn_output_step_{self.num_timesteps}.pt")
                )
                print(f"Saved CNN outputs at timestep {self.num_timesteps}.")
                self.logger.info(f"Saved CNN outputs at timestep {self.num_timesteps}.")
            else:
                print("No observation sample provided to extract features.")
        return True

    def _extract_cnn_features(self, observations):
        cnn_model = self.model.policy.features_extractor  # Access the custom CNN
        with torch.no_grad():
            obs_tensor = torch.tensor(observations, dtype=torch.float32, device=self.model.device)
            # Forward pass through the CNN to get both processed output and raw features
            raw_features, processed_output = cnn_model.get_intermediate_features(obs_tensor)
        return processed_output, raw_features      
