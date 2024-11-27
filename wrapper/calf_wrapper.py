import numpy as np
from gymnasium import Wrapper


class CALFNominalWrapper():
    def __init__(self, controller):
        self.controller = controller

    def compute_action(self, observation):
        action = ...
        return action 


class CALFEnergyPendulumWrapper(CALFNominalWrapper):
    def compute_action(self, observation):
        cos_theta, _, angular_velocity = observation
        return self.controller.compute(cos_theta, angular_velocity)


class CALFWrapper(Wrapper):
    def __init__(self, 
                 env, 
                 fallback_policy: CALFNominalWrapper = None, 
                 calf_decay_rate: float = 0.0005,
                 initial_relax_prob: float = 0.5,
                 relax_prob_base_step_factor: float = 0.9,
                 relax_prob_episode_factor: float = 0.9):
        super().__init__(env)
        self.calf_value = None
        self.fallback_policy = fallback_policy
        self.calf_activated_count = 0
        self.calf_decay_rate = calf_decay_rate

        # Relax prob at the 1st episode and 1st step
        self.relax_prob_base_step_factor = relax_prob_base_step_factor

        # Factor to update relax_prob_step_factor at the end of each episode
        self.relax_prob_episode_factor = relax_prob_episode_factor

        # Intiated with base_step_factor
        # and dropped after each episode (episode to episode)
        self.relax_prob_step_factor = relax_prob_base_step_factor

        # Actual relax prob
        self.initial_relax_prob = initial_relax_prob
        self.relax_prob = self.initial_relax_prob

        # Only activate after 1st episode
        self.relax_prob_episode_activated = False
    
    def update_current_value(self, value, step):
        self.current_value = value
        self.current_step_n = step

    def is_calf_value_satisfied(self):
        is_decay = False

        if self.calf_value is None:
            self.calf_value = self.current_value
        elif self.current_value - self.calf_value > self.calf_decay_rate:
            is_decay = True
            self.calf_activated_count += 1
            self.calf_value = self.current_value

        return is_decay
    
    def is_calf_value_decay(self):
        eps = np.random.random()
        self.relax_prob = self.relax_prob * self.relax_prob_step_factor ** self.current_step_n

        if eps < self.relax_prob or \
              self.is_calf_value_satisfied():
            return True
        return False

    def step(self, action):
        if not self.relax_prob_episode_activated:
            self.relax_prob_episode_activated = True
        # print(f"Action: {action}")

        if not hasattr(self, "current_value"):
            raise Exception("No current_value found")
        
        if self.is_calf_value_decay():
            calf_action = action.copy()
            
        else:
            if self.fallback_policy is None:
                calf_action = np.zeros_like(action)
            else:
                calf_action = self.fallback_policy.compute_action(self.last_obs)
            
        obs, reward, terminated, truncated, info = self.env.step(calf_action)
        reward = float(reward)  # Ensure reward is a scalar
        
        self.last_obs = obs
        # Log observation, reward, and done flags
        # print(f"Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        print(f"Resetting environment with args: {kwargs}")
        print(f"Resetting environment last calf_activated_count: {self.calf_activated_count}")

        if self.relax_prob_episode_activated:
            self.relax_prob_step_factor = self.relax_prob_base_step_factor * \
                                          self.relax_prob_episode_factor
        self.relax_prob = self.initial_relax_prob
        self.calf_value = None
        self.calf_activated_count = 0
        self.last_obs, info = self.env.reset(**kwargs)
        return self.last_obs, info
    