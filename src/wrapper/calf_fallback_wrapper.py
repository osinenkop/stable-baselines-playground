
import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

from src.wrapper.calf_wrapper import CALFNominalWrapper


class CALFEnergyPendulumWrapper(CALFNominalWrapper):
    """
    This class inherits from CALFWrapper and integrates with the EnergyBasedController. 
    It is designed to operate with EnergyBasedController as the primary fallback mechanism within the CALFWrapper structure. 
    Upon initialization, an instance of the EnergyBasedController must be provided, ensuring seamless energy-based fallback functionality and robust handling of controller logic.
    """
    def compute_action(self, observation):
        cos_angle, sin_angle, angular_velocity = observation
        angle = np.arctan2(sin_angle, cos_angle)

        control_action = self.controller.compute(angle, cos_angle, angular_velocity)
        
        # Clip the action to the valid range for the Pendulum environment
        return np.clip([control_action], -2.0, 2.0)
        # return control_action


class CALFPPOPendulumWrapper(CALFNominalWrapper):
    """
    This class inherits from CALFWrapper and utilizes a pre-trained PPO checkpoint as the fallback mechanism. 
    During initialization, the class requires the path to a checkpoint of a trained PPO model, 
    which it integrates as a fallback within the CALFWrapper structure. 
    """
    def __init__(self, checkpoint_path, action_low, action_high, device=None):
        self.model = PPO.load(checkpoint_path)
        if device is None:
            self.device = "cuda" if th.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.action_space_low = action_low
        self.action_space_high = action_high
    
    def compute_action(self, observation):
        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(observation, self.device)
            actions, _, _ = self.model.policy(obs_tensor)
            actions = actions.cpu().numpy()

        return np.clip(actions, self.action_space_low, self.action_space_high)
