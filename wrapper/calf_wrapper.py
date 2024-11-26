import numpy as np
from gymnasium import Wrapper


class CALFWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_value = None
    
    def is_filter_activated(self):
        if self.last_value is None:
            self.last_value = self.current_value
            return False
        else:
            ret = False
            if self.current_value - self.last_value > 0.0005:
                ret = True
                print("Activated")

                self.last_value = self.current_value
            return ret

    def step(self, action):
        print(f"Action: {action}")

        if not hasattr(self, "current_value"):
            raise Exception("No current_value found")
        
        if self.is_filter_activated():
            chosen_action = np.zeros_like(action)
        else:
            chosen_action = action
            
        obs, reward, terminated, truncated, info = self.env.step(chosen_action)
        reward = float(reward)  # Ensure reward is a scalar
        
        # Log observation, reward, and done flags
        print(f"Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        print(f"Resetting environment with args: {kwargs}")
        self.last_value = None
        return self.env.reset(**kwargs)