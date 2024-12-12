import os
from stable_baselines3.common.callbacks import BaseCallback

class GradientMonitorCallback(BaseCallback):
    """
    Custom callback to monitor the gradients of the CNN during training and save them for visualization.
    """
    def __init__(self, log_file="logs/gradients_log.csv", verbose=0):
        super().__init__(verbose)
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def _on_step(self) -> bool:
        # Open file in append mode
        with open(self.log_file, "a") as f:
            # Access the feature extractor (CNN) parameters
            for name, param in self.model.policy.features_extractor.named_parameters():
                if param.requires_grad:
                    grad_mean = param.grad.mean().item() if param.grad is not None else 0
                    grad_std = param.grad.std().item() if param.grad is not None else 0
                    # Write to log
                    f.write(f"{self.num_timesteps},{name},{grad_mean},{grad_std}\n")
        return True

