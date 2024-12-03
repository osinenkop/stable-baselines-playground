import numpy as np
import torch as th

from gymnasium import Wrapper
from stable_baselines3.common.logger import configure
from copy import copy


class CALFNominalWrapper():
    def __init__(self, controller):
        self.controller = controller

    def compute_action(self, observation):
        action = ...
        return action 


class CALFEnergyPendulumWrapper(CALFNominalWrapper):
    def compute_action(self, observation):
        cos_angle, sin_angle, angular_velocity = observation
        angle = np.arctan2(sin_angle, cos_angle)

        control_action = self.controller.compute(angle, cos_angle, angular_velocity)
        
        # Clip the action to the valid range for the Pendulum environment
        return np.clip([control_action], -2.0, 2.0)
        # return control_action


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
        self.debug = kwargs.get("debug", False)

        self.policy_model = None

    def copy_policy_model(self, policy_model):
        self.policy_model = copy(policy_model)
    
    # Ignore this function
    def update_current_value(self, value, step):
        # Receive state-value from agent
        self.current_value = value
        self.current_step_n = step

    def get_state_value(self, state):
        with th.no_grad():
            return self.policy_model.predict_values(
                self.policy_model.obs_to_tensor(state)[0]
            )

    def is_calf_value_decay(self, current_state):
        is_decay = False
        current_value = self.get_state_value(current_state)

        # In case before PPO starts learning
        if current_value is None:
            return is_decay

        if self.calf_value is None:
            self.calf_value = current_value

        # V̂_w (st) − V̂_w(s†) ≥ ν̄
        elif current_value - self.calf_value >= self.calf_decay_rate:
            is_decay = True
            self.calf_decay_count += 1

            # store V̂_w(s†)
            self.calf_value = current_value
        
        ## DEBUG {
        if self.debug:
            if is_decay:
                print("[DEBUG]: Line 6 Passed")
            else:
                print("[DEBUG]: Line 6 Fallback")
        # }
        
        return is_decay
    
    def get_calf_action(self):
        return self.calf_action if hasattr(self, "calf_action") \
                                else self.fallback_policy.compute_action(self.calf_state)
    
    def is_agent_action_perform(self, current_state):
        eps = np.random.random()
        
        self.logger.record("calf/last_relax_prob", self.relax_prob)

        self.debug and print("[DEBUG]: Line 11")
        if eps < self.relax_prob or \
              self.is_calf_value_decay(current_state):
            self.calf_activated_count += 1
            return True
        return False

    def update_calf_action(self, agent_action, current_state):
        if self.is_agent_action_perform(current_state):
            self.debug and print("[DEBUG]: Line 12")
            self.calf_action = agent_action.copy()
            
        else:
            if self.fallback_policy is None:
                self.calf_action = np.zeros_like(agent_action)
            else:
                self.debug and print("[DEBUG]: Line 14")
                self.calf_action = self.fallback_policy.compute_action(current_state)

    def step(self, action):
        self.debug and print("[DEBUG]: Line 5")
        # At step 0, self.calf_state was received from reset
        obs, reward, terminated, truncated, info = self.env.step(
            self.get_calf_action()
        )

        if not self.relax_prob_episode_activated:
            self.relax_prob_episode_activated = True

        if not hasattr(self, "current_value"):
            raise Exception("No current_value found")
        
        self.update_calf_action(action, obs)
        
        reward = float(reward)  # Ensure reward is a scalar
        
        self.debug and print("[DEBUG]: Line 16")
        self.relax_prob = np.clip(self.relax_prob * self.relax_prob_step_factor,
                                  0, 1)
        
        # Log observation, reward, and done flags
        # print(f"Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        
        return obs, reward, terminated, truncated, info
    
    def reset_internal_params(self):
        self.logger.record("calf/calf_value", self.calf_value)
        self.logger.record("calf/current_value", self.current_value)
        self.logger.record("calf/calf_decay_count", self.calf_decay_count)
        self.logger.record("calf/calf_activated_count", self.calf_activated_count)

        if self.relax_prob_episode_activated:
            self.relax_prob_step_factor = self.relax_prob_base_step_factor
            self.initial_relax_prob = np.clip(
                self.initial_relax_prob + self.initial_relax_prob * self.relax_prob_episode_factor,
                0, 1)
            self.relax_prob = self.initial_relax_prob

        self.calf_value = None
        self.calf_activated_count = 0
        self.calf_decay_count = 0

        self.logger.record("calf/init_relax_prob", self.relax_prob)
        # print(f"Resetting environment last self.relax_prob: {self.relax_prob}")

    def reset(self, **kwargs):
        # print(f"Resetting environment with args: {kwargs}")
        
        self.reset_internal_params()
        self.calf_state, info = self.env.reset(**kwargs)
        return self.calf_state, info
    