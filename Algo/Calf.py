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
    def __init__(self, replacing_probability = 0.2, best_value_local = None):

        self.replacing_probability = replacing_probability

        self.best_policy_global = None
        self.best_value_local = -math.inf
        self.best_value_global = -math.inf
        self.avr_reward = -math.inf
        self.nu = 0.02 # 1e-5
        self.calf_filter_delay = 5
        print("CALFQ Filter init")

    def init_policy(self, policy):
        self.best_policy_global = deepcopy(policy)
        self.best_policy_weights = policy.state_dict()
        self.best_policy_global.load_state_dict(self.best_policy_weights)

    
    def update_global_policy(self):
        # self.best_policy_global.load_state_dict(self.best_policy_weights)
        self.best_value_global = self.best_value_local
        print(style.RED, "Global best policy updated", style.RESET)

    def sampling_time_init(self, sampling_time):
        self.sampling_time = sampling_time
        self.nu = self.nu * sampling_time

    def value_reset(self): 
        self.best_value_local = -math.inf
        self.best_value_global = -math.inf
    
    def get_last_good_model(self):
        return self.best_policy_global
    
    def update_flag(self):
        return 0

    def compute_action(self, action, observation, 
                       best_value_local, Q_value, 
                       current_policy, obs_tensor,
                       iteration):
        
        self.iteration = iteration
 
        if iteration < self.calf_filter_delay:
            return action
        else:
            if best_value_local - Q_value <= self.nu:
                self.best_value_local = Q_value
                return action
            if (np.random.random() >= self.replacing_probability):
                return action
            
            # print(style.RED, "Replace Action", style.RESET)
            action, _, _ = self.best_policy_global(obs_tensor)
            return action