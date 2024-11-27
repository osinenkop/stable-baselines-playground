import numpy as np
from gymnasium import Wrapper


class CALFWrapper(Wrapper):
    def __init__(self, 
                 env, 
                 fallback_policy=None, 
                 init_relax_prob=0.5,
                 calf_decay_rate=0.0005):
        super().__init__(env)
        self.calf_value = None
        self.fallback_policy = fallback_policy
        self.calf_activated_count = 0
        self.relax_prob = init_relax_prob
        self.calf_decay_rate = calf_decay_rate
    
    def update_values(self, value):
        self.current_value = value

    def is_calf_value_decay(self):
        is_decay = False

        if self.calf_value is None:
            self.calf_value = self.current_value
        elif self.current_value - self.calf_value > self.calf_decay_rate:
            is_decay = True
            self.calf_activated_count += 1
            self.calf_value = self.current_value

        return is_decay

    def step(self, action):
        # print(f"Action: {action}")

        if not hasattr(self, "current_value"):
            raise Exception("No current_value found")
        
        if self.is_calf_value_decay():
            chosen_action = action.copy()
            
        else:
            if self.fallback_policy is None:
                chosen_action = np.zeros_like(action)
            else:
                cos_theta, _, angular_velocity = self.last_obs
                chosen_action = self.fallback_policy.compute(cos_theta, angular_velocity)
            
        obs, reward, terminated, truncated, info = self.env.step(chosen_action)
        reward = float(reward)  # Ensure reward is a scalar
        
        self.last_obs = obs
        # Log observation, reward, and done flags
        # print(f"Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        print(f"Resetting environment with args: {kwargs}")
        print(f"Resetting environment last calf_activated_count: {self.calf_activated_count}")
        self.calf_value = None
        self.calf_activated_count = 0
        self.last_obs, info = self.env.reset(**kwargs)
        return self.last_obs, info
    