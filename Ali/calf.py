import numpy as np

from copy import deepcopy

from controllers.energybased import EnergyBasedController

class CaLF():
    def __init__(self, mu = 1e-6, replacing_probability = 0.8, best_policy = None):
        self.replacing_probability = replacing_probability
        self.policy_update = True
        self.mu = mu
        self.best_policy = best_policy
        self.inital_policy = best_policy
        self.nominal_policy = EnergyBasedController()

    def update(self, policy):
        self.best_policy = deepcopy(policy)
        if self.inital_policy is None:
            self.inital_policy = deepcopy(policy)
            self.policy_update = True

    def reset(self):
        self.best_policy = deepcopy(self.inital_policy)

    def pass_or_replace(self, action, obs, Q_value_best, Q_value, policy):
        if Q_value_best[-1].cpu().detach().numpy()[-1] - Q_value[-1].cpu().detach().numpy()[-1] > self.mu:
            self.policy_update = False
            if np.random.random() <= self.replacing_probability:
                cos_theta, _, angular_velocity = obs
                new_action = self.nominal_policy.compute(cos_theta,angular_velocity)
                new_action = [np.clip([new_action], -2, 2)]
                return new_action
        else:
            self.update(policy)
        
        return action