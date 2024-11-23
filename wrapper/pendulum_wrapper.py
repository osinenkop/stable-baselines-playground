import numpy as np
import gymnasium as gym
import cv2
import matplotlib.pyplot as plt

from gymnasium.spaces import Box
from gymnasium import ObservationWrapper

class LoggingWrapper(gym.Wrapper):
    def step(self, action):
        print(f"Action: {action}")
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)  # Ensure reward is a scalar
        
        # Log observation, reward, and done flags
        print(f"Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        print(f"Resetting environment with args: {kwargs}")
        return self.env.reset(**kwargs)

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