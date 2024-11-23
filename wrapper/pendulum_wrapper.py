import numpy as np
import gymnasium as gym
import cv2
import matplotlib.pyplot as plt

from gymnasium.spaces import Box
from gymnasium import ObservationWrapper

# Wrap to ensure the observation space has channels last
class EnsureChannelsLastWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space

        if len(obs_space.shape) == 3:  # (C, H, W)
            # Single frame: (C, H, W) -> (H, W, C)
            self.observation_space = Box(
                low=obs_space.low.transpose(1, 2, 0),
                high=obs_space.high.transpose(1, 2, 0),
                shape=(obs_space.shape[1], obs_space.shape[2], obs_space.shape[0]),
                dtype=obs_space.dtype,
            )
        elif len(obs_space.shape) == 4:  # (stack, C, H, W)
            # Stacked frames: (stack, C, H, W) -> (stack, H, W, C)
            self.observation_space = Box(
                low=obs_space.low.transpose(0, 2, 3, 1),
                high=obs_space.high.transpose(0, 2, 3, 1),
                shape=(obs_space.shape[0], obs_space.shape[2], obs_space.shape[3], obs_space.shape[1]),
                dtype=obs_space.dtype,
            )

    def observation(self, observation):
        if observation.ndim == 3:  # Single frame (C, H, W) -> (H, W, C)
            return observation.transpose(1, 2, 0)
        elif observation.ndim == 4:  # Stacked frames (stack, C, H, W) -> (stack, H, W, C)
            return observation.transpose(0, 2, 3, 1)
        return observation

class NormalizeObservation(ObservationWrapper):
    def __init__(self, env):
        super(NormalizeObservation, self).__init__(env)
        # Modify observation space to reflect normalization
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

    def observation(self, observation):
        normalized_obs = observation / 255.0  # Example normalization logic
        return normalized_obs

class ResizeObservation(ObservationWrapper):
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shape[0], shape[1], 3), dtype=np.uint8
        )

    def observation(self, observation):
        # Debug: Check if the observation is empty or not
        if observation is None or observation.size == 0:
            print("Error: Observation is empty or not properly generated.")
            raise ValueError("Observation is empty or not properly generated.")

        # Resize the observation using OpenCV
        resized_observation = cv2.resize(observation, (self.shape[1], self.shape[0]))

        return resized_observation