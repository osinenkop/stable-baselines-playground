import os
import torch
import numpy as np
import matplotlib.pyplot as plt

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

            # print(f"Rollout buffer first observation stats: Min={self.model.rollout_buffer.observations[0].min()}, "
            #     f"Max={self.model.rollout_buffer.observations[0].max()}, Shape={self.model.rollout_buffer.observations[0].shape}")

            # print(f"Rollout buffer last (most recent) observation stats: Min={self.model.rollout_buffer.observations[-1].min()}, "
            #     f"Max={self.model.rollout_buffer.observations[-1].max()}, Shape={self.model.rollout_buffer.observations[-1].shape}")

            # Fetch the most recent observation from the rollout buffer
            try:
                obs_sample = self.model.rollout_buffer.observations[self.every_n_steps // 2]
            except AttributeError:
                print("Rollout buffer not found, skipping CNN saving.")
                return True             

            # print(f"self.model.rollout_buffer.observations.shape = {self.model.rollout_buffer.observations.shape}")

            # print(f"self.model.rollout_buffer.observations: Min={self.model.rollout_buffer.observations.min()}, Max={self.model.rollout_buffer.observations.max()}")

            # print(f"Observation from rollout_buffer: Min={obs_sample.min()}, Max={obs_sample.max()}, Shape={obs_sample.shape}")

            # Ensure the observation is properly formatted and permuted
            if isinstance(obs_sample, np.ndarray):  # If using NumPy arrays
                # print(f"Before tensor conversion: Min={obs_sample.min()}, Max={obs_sample.max()}, Shape={obs_sample.shape}")
                obs_sample = torch.tensor(obs_sample, dtype=torch.float32).to(self.model.device)
                # print(f"After tensor conversion: Min={obs_sample.min()}, Max={obs_sample.max()}, Shape={obs_sample.shape}")
            elif isinstance(obs_sample, torch.Tensor):  # If using Torch tensors
                obs_sample = obs_sample.to(self.model.device)

            # image = obs_sample[0].permute(1, 2, 0).cpu().numpy()
            # image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0.0, 1.0]
            # plt.imshow(image)
            # plt.title("Image from PendulumVisual as perceived by CNN in callback")
            # plt.axis('off')  # Hide axes
            # plt.show()  # Block execution until the plot is closed 

            # Permute the observation tensor to match PyTorch's expected input shape (N, C, H, W)
            # obs_sample = obs_sample.permute(0, 3, 1, 2)

            # print(f"Observation sent to CNN: Min={obs_sample.min()}, Max={obs_sample.max()}, Shape={obs_sample.shape}")

            # Pass the observations through the CNN
            cnn_model = self.model.policy.features_extractor  # Access the custom CNN
            with torch.no_grad():
                layer_features = cnn_model.get_layer_features(obs_sample)

            # Save the features
            torch.save(
                layer_features,
                os.path.join(self.save_path, f"cnn_layer_features_step_{self.num_timesteps}.pt")
            )
            print(f"Saved CNN outputs at timestep {self.num_timesteps}.")
        return True

    # def _on_rollout_end(self) -> None:
    #     print("Saving CNN output at rollout end...")
    #     obs_sample = self.model.rollout_buffer.observations[-1]

    #     # Ensure the observation is properly formatted and permuted
    #     if isinstance(obs_sample, np.ndarray):  # If using NumPy arrays
    #         # print(f"Before tensor conversion: Min={obs_sample.min()}, Max={obs_sample.max()}, Shape={obs_sample.shape}")
    #         obs_sample = torch.tensor(obs_sample, dtype=torch.float32).to(self.model.device)
    #         # print(f"After tensor conversion: Min={obs_sample.min()}, Max={obs_sample.max()}, Shape={obs_sample.shape}")
    #     elif isinstance(obs_sample, torch.Tensor):  # If using Torch tensors
    #         obs_sample = obs_sample.to(self.model.device)

    #     image = obs_sample[0].permute(1, 2, 0).cpu().numpy()
    #     image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0.0, 1.0]
    #     plt.imshow(image)
    #     plt.title("Image from PendulumVisual as perceived by CNN in callback")
    #     plt.axis('off')  # Hide axes
    #     plt.show()  # Block execution until the plot is closed 

    #     print(f"Observation sent to CNN: Min={obs_sample.min()}, Max={obs_sample.max()}, Shape={obs_sample.shape}")

    #     # Pass the observations through the CNN
    #     cnn_model = self.model.policy.features_extractor  # Access the custom CNN
    #     with torch.no_grad():
    #         layer_features = cnn_model.get_layer_features(obs_sample)

    #     # Save the features
    #     torch.save(
    #         layer_features,
    #         os.path.join(self.save_path, f"cnn_layer_features_step_{self.num_timesteps}.pt")
    #     )
    #     print(f"Saved CNN outputs at timestep {self.num_timesteps}.")
    #     return True

    def _extract_cnn_features(self, observations):
        cnn_model = self.model.policy.features_extractor  # Access the custom CNN
        with torch.no_grad():
            # Permute the observation tensor to match PyTorch's format (N, C, H, W)
            obs_tensor = torch.tensor(observations, dtype=torch.float32, device=self.model.device).permute(0, 3, 1, 2)
            layer_features = cnn_model.get_layer_features(obs_tensor)
        return layer_features
