import numpy as np
import math

from copy import deepcopy

class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

class calfq_filter():
    def __init__(self, replacing_probability = 0.5, best_value_local = None):

        self.replacing_probability = replacing_probability

        self.best_policy = None
        self.best_value_local = -math.inf
        self.best_value_global = -math.inf
        self.nu = 0.01 #1e-5
        self.calf_filter_delay = 2
        print("CALFQ Filter init")

    def reset(self):
        self.best_value_local = -math.inf
        self.best_value_global = -math.inf
        self.best_policy = None
        print("CALFQ Filter reset")

    def init_policy(self, policy):
        self.best_policy_global = deepcopy(policy)
        self.best_policy = deepcopy(policy)

    def update_global_policy(self):
        if self.iteration < self.calf_filter_delay:
            pass
        else:
            print(style.CYAN, "Best_value_global = ", self.best_value_global, style.RESET)
            print("Best_value_local = ",self.best_value_local )
            # if (self.best_value_global == -math.inf) or (self.best_value_global < self.best_value_local):
            if (self.best_value_global == -math.inf) or ((self.best_value_local - self.best_value_global) >= (self.nu)):
                self.best_policy_global = deepcopy(self.best_policy)
                self.best_value_global = self.best_value_local
                print(style.RED, "Global best policy updated", style.RESET)

    def sampling_time_init(self, sampling_time):
        self.sampling_time = sampling_time
        self.nu = self.nu * sampling_time
    
    def get_last_good_model(self):
        return self.best_policy_global
    
    def update_flag(self):
        return 0

    def compute_action(self, action, observation, 
                       best_value_local, Q_value, 
                       current_policy, obs_tensor,
                       iteration):
        self.iteration = iteration
        # print(style.RED, Q_value, style.RESET)
        # print(style.CYAN, best_value_local, style.RESET)
        if iteration < self.calf_filter_delay:
            return action
        else:
            if best_value_local <= Q_value:
            # if ( Q_value - best_value_local ) >= (self.nu):
                # print(style.RED, "Best_value_local updated", style.RESET)
                self.best_value_local = Q_value
                self.best_policy = deepcopy(current_policy)
                return action
            if (np.random.random()<=self.replacing_probability):
                # print(style.CYAN, "Constraints failed... Probability of relax applied", style.RESET)
                return action
            # print(style.CYAN, "Constraints failed... Global best policy applied", style.RESET)
            action, _, _ = self.best_policy_global(obs_tensor)
            return action