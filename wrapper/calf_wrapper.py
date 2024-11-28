import numpy as np
from gymnasium import Wrapper
from stable_baselines3.common.logger import configure


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
                 relax_prob_episode_factor: float = 0.1,
                 **kwargs):
        super().__init__(env)
        self.calf_value = None
        self.current_value = None
        self.fallback_policy = fallback_policy
        self.calf_activated_count = 0
        self.calf_decay_count = 0
        self.calf_decay_rate = calf_decay_rate

        # Relax prob at the 1st episode and 1st step
        self.relax_prob_base_step_factor = relax_prob_base_step_factor

        # Factor to update relax_prob_step_factor at the end of each episode
        self.relax_prob_episode_factor = relax_prob_episode_factor

        # Intiated with base_step_factor
        # and increase after each episode (episode to episode)
        self.relax_prob_step_factor = relax_prob_base_step_factor

        # Actual relax prob
        self.initial_relax_prob = initial_relax_prob
        self.relax_prob = self.initial_relax_prob

        # Only activate after 1st episode
        self.relax_prob_episode_activated = False

        self.logger = kwargs.get("logger", configure())

    
    def update_current_value(self, value, step):
        self.current_value = value
        self.current_step_n = step

    def is_calf_value_decay(self):
        is_decay = False

        # In case before PPO starts learning
        if self.current_value is None:
            return is_decay

        if self.calf_value is None:
            self.calf_value = self.current_value
        elif self.current_value - self.calf_value > self.calf_decay_rate:
        # elif self.calf_value - self.current_value > self.calf_decay_rate:
            is_decay = True
            self.calf_decay_count += 1
            self.calf_value = self.current_value
        
        ## DEBUG {
        # if is_decay:
            # print("[DEBUG]: Line 6 Passed")
        # else:
            # print("[DEBUG]: Line 6 Fallback")
        ## }
        
        return is_decay
    
    def is_agent_action_perform(self):
        eps = np.random.random()
        
        self.logger.record("calf/last_relax_prob", self.relax_prob)
        if eps < self.relax_prob or \
              self.is_calf_value_decay():
            self.calf_activated_count += 1
            return True
        return False

    def update_calf_action(self, agent_action, calf_state):
        if self.is_agent_action_perform():
            # print("[DEBUG]: Line 12")
            self.calf_action = agent_action.copy()
            
        else:
            if self.fallback_policy is None:
                self.calf_action = np.zeros_like(agent_action)
            else:
                # print("[DEBUG]: Line 13")
                self.calf_action = self.fallback_policy.compute_action(calf_state)

    def step(self, action):
        # print("[DEBUG]: Line 5")
        
        obs, reward, terminated, truncated, info = self.env.step(
            getattr(self, 
                    "calf_action", 
                    self.fallback_policy.compute_action(self.calf_state))
        )

        if not self.relax_prob_episode_activated:
            self.relax_prob_episode_activated = True

        if not hasattr(self, "current_value"):
            raise Exception("No current_value found")
        
        self.update_calf_action(action, obs)
        
        reward = float(reward)  # Ensure reward is a scalar
        
        # print("[DEBUG]: Line 14", self.current_step_n)
        self.relax_prob = np.clip(self.relax_prob * self.relax_prob_step_factor,
                                  0, 1)
        
        # Log observation, reward, and done flags
        # print(f"Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        
        return obs, reward, terminated, truncated, info
    
    def reset_internal_params(self):
        self.logger.record("calf/init_relax_prob", self.relax_prob)
        self.logger.record("calf/calf_value", self.calf_value)
        self.logger.record("calf/current_value", self.current_value)
        self.logger.record("calf/calf_decay_count", self.calf_decay_count)
        self.logger.record("calf/calf_activated_count", self.calf_activated_count)

        if self.relax_prob_episode_activated:
            self.relax_prob_step_factor = self.relax_prob_base_step_factor
            self.relax_prob = np.clip(
                self.relax_prob + self.relax_prob * self.relax_prob_episode_factor,
                0, 1)

        self.calf_value = None
        self.calf_activated_count = 0
        print(f"Resetting environment last self.relax_prob: {self.relax_prob}")

    def reset(self, **kwargs):
        print(f"Resetting environment with args: {kwargs}")
        
        self.reset_internal_params()
        self.calf_state, info = self.env.reset(**kwargs)
        return self.calf_state, info
    