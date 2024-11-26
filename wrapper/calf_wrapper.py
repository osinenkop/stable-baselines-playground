import numpy as np
from gymnasium import Wrapper


class CALFWrapper(Wrapper):
    def __init__(self, env, nominal_controller=None):
        super().__init__(env)
        self.last_value = None
        self.nominal_controller = nominal_controller
        self.calf_activated_count = 0
    
    def is_filter_activated(self):
        if self.last_value is None:
            self.last_value = self.current_value
            return False
        else:
            ret = False
            if self.current_value - self.last_value > 0.0005:
                ret = True
                self.calf_activated_count += 1
                self.last_value = self.current_value
            return ret

    def step(self, action):
        # print(f"Action: {action}")

        if not hasattr(self, "current_value"):
            raise Exception("No current_value found")
        
        if self.is_filter_activated():
            chosen_action = action
            
        else:
            if self.nominal_controller is None:
                chosen_action = np.zeros_like(action)
            else:
                cos_theta, _, angular_velocity = self.last_obs
                chosen_action = self.nominal_controller.compute(cos_theta, angular_velocity)
            
        obs, reward, terminated, truncated, info = self.env.step(chosen_action)
        reward = float(reward)  # Ensure reward is a scalar
        
        self.last_obs = obs
        # Log observation, reward, and done flags
        # print(f"Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        print(f"Resetting environment with args: {kwargs}")
        print(f"Resetting environment last calf_activated_count: {self.calf_activated_count}")
        self.last_value = None
        self.calf_activated_count = 0
        self.last_obs, info = self.env.reset(**kwargs)
        return self.last_obs, info