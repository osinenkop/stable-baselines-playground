import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
import argparse
import numpy as np
import time
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit
from mygym.my_pendulum import PendulumRenderFix
# Import the custom callback from callback.py
from callback.plotting_callback import PlottingCallback
from stable_baselines3.common.utils import get_linear_fn
from controller.pid import PIDController
from controller.energybased import EnergyBasedController


from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from copy import deepcopy

from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule


SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")

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

class calfq():
    def __init__(self, replacing_probability = 0.5, best_Q_value = None):

        self.critic_struct = 'quad-mix'
        self.observation_target = []


        self.replacing_probability = replacing_probability
        self.best_Q_value = best_Q_value

        self.best_policy = None


        self.action_LG = []
        self.observation_LG = []
        self.w_critic_LG = []

        self.nu = 1e-5
        

    def first_lg_init(self, action, observation, w_critic):
        self.action_LG = action
        self.observation_LG = observation
        self.w_critic_LG = w_critic

    def sampling_time_init(self, sampling_time):
        self.sampling_time = sampling_time
        self.nu = self.nu * sampling_time

    def _critic(self, observation, action, w_critic):
        """
        Critic a.k.a. objective learner: a routine that models something related to the objective, e.g., value function, Q-function, advantage etc.
        
        Currently, this implementation is for linearly parametrized models.

        """
        def uptria2vec(mat):
            """
            Convert upper triangular square sub-matrix to column vector.
            
            """    
            n = mat.shape[0]
            
            vec = np.zeros( (int(n*(n+1)/2)) )
            
            k = 0
            for i in range(n):
                for j in range(i, n):
                    vec[k] = mat[i, j]
                    k += 1
                    
            return vec

        if self.observation_target == []:
            chi = np.concatenate([observation, action])
        else:
            chi = np.concatenate([observation - self.observation_target, action])
        
        # regressor_critic = np.concatenate([uptria2vec( np.outer(chi**2, chi**2) ), uptria2vec( np.outer(chi**2, chi) ), uptria2vec( np.outer(chi, chi) )])
        # print(regressor_critic.size)

        if self.critic_struct == 'quad-lin':
            regressor_critic = np.concatenate([ uptria2vec( np.outer(chi, chi) ), chi ])
        elif self.critic_struct == 'quadratic':
            regressor_critic = np.concatenate([ uptria2vec( np.outer(chi, chi) ) ])   
        elif self.critic_struct == 'quad-nomix':
            regressor_critic = chi * chi
        elif self.critic_struct == 'quad-mix':
            regressor_critic = np.concatenate([ observation**2, np.kron(observation, action), action**2 ]) 
        elif self.critic_struct == 'poly3':
            regressor_critic = np.concatenate([uptria2vec( np.outer(chi**2, chi) ), uptria2vec( np.outer(chi, chi) )])
        elif self.critic_struct == 'poly4':               
            regressor_critic = np.concatenate([uptria2vec( np.outer(chi**2, chi**2) ), 
                                               uptria2vec( np.outer(chi**2, chi) ), 
                                               uptria2vec( np.outer(chi, chi) )])

        return w_critic @ regressor_critic

    def CALF_constr(self, action, observation, w_critic):
        critic_new = self._critic(observation, action, w_critic)                            # Q^w (x_k, u_k)
        critic_LG = self._critic(self.observation_LG, self.action_LG, self.w_critic_LG)     # Q^w°(x°, u°)
        return critic_new - critic_LG                                                       # Q^w (x_k, u_k) - Q^w°(x°, u°)

    def pass_or_replace(self, action, observation, w_critic):

        if self.action_LG == []:
            self.first_lg_init(action, observation, w_critic)

        res = self.CALF_constr(action, observation, w_critic)

        if res < (-self.nu):
            new_action = action
            self.w_critic_LG = w_critic
            self.observation_LG = observation
            self.action_LG = action
        else:
            pass

class calfq_filter():
    def __init__(self, replacing_probability = 0.5, Q_value_lg = None):

        self.replacing_probability = replacing_probability
        self.Q_value_lg = Q_value_lg

        self.best_policy = None

        self.nu = 1e-5
        print("CALFQ Filter init")

    def init_policy(self, policy):
        self.best_policy = deepcopy(policy)

    def sampling_time_init(self, sampling_time):
        self.sampling_time = sampling_time
        self.nu = self.nu * sampling_time

    def value_reset(self):
        self.Q_value_lg = None
    
    def get_last_good_model(self):
        return self.best_policy

    def compute_action(self, action, observation, last_good_Q_value, Q_value, current_policy, obs_tensor):

        if (last_good_Q_value) == None or ((last_good_Q_value - Q_value) >= (self.nu)) or (np.random.random()<=self.replacing_probability): # or randomize
            self.best_policy = deepcopy(current_policy)
            self.last_good_Q_value = Q_value
            return action
        else:
            action, _, _ = self.best_policy(obs_tensor)
            return action

def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        calf_filter.value_reset()
        self.policy = calf_filter.get_last_good_model()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # TODO Add CaLF here
            
            clipped_actions = self.calf_filter.compute_action(clipped_actions, self._last_obs[-1], 
                                                               self.calf_filter.best_policy.predict_values(obs_tensor), 
                                                               self.policy.predict_values(obs_tensor),
                                                               self.policy,
                                                               obs_tensor)
            
            if isinstance(clipped_actions, torch.Tensor):
                clipped_actions = clipped_actions.cpu().detach().numpy()
            else:
                clipped_actions = np.array(clipped_actions)

            # TODO Update CaLF foreach episode
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # if dones[-1] is True:
            #     if self.calf_filter.policy_update is True:
            #         self.calf_filter.update(self.policy)
            #     else:
            #         self.calf_filter.policy_update = True
            #         print("Policy Updated")

            self.num_timesteps += env.num_envs
            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True   

def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()

        callback.on_training_end()

        return self



# Initialize the argument parser
parser = argparse.ArgumentParser(description="PPO Training and Evaluation for Pendulum")
parser.add_argument("--notrain", action="store_true", help="Skip the training phase")

# Parse the arguments
args = parser.parse_args()

matplotlib.use("TkAgg")  # Try "Qt5Agg" if "TkAgg" doesn't work

# Register the environment
gym.envs.registration.register(
    id="PendulumRenderFix-v0",
    entry_point="mygym.my_pendulum:PendulumRenderFix",
)

# Use your custom environment for training
env = gym.make("PendulumRenderFix-v0")

env = TimeLimit(env, max_episode_steps=1000)  # Set a maximum number of steps per episode

# ---------------------------------
# Initialize the PID controller
kp = 5.0  # Proportional gain
ki = 0.1   # Integral gain
kd = 1.0   # Derivative gain
pid = PIDController(kp, ki, kd, setpoint=0.0)  # Setpoint is the upright position (angle = 0)

dt = 0.05  # Action time step for the simulation
# ---------------------------------
# Initialize the energy-based controller
controller = EnergyBasedController()
# ---------------------------------

total_timesteps = 500000

# Define the hyperparameters for PPO
ppo_hyperparams = {
    "learning_rate": 5e-4,  # The step size used to update the policy network. Lower values can make learning more stable.
    "n_steps": 4000,  # Number of steps to collect before performing a policy update. Larger values may lead to more stable updates.
    "batch_size": 200,  # Number of samples used in each update. Smaller values can lead to higher variance, while larger values stabilize learning.
    "gamma": 0.98,  # Discount factor for future rewards. Closer to 1 means the agent places more emphasis on long-term rewards.
    "gae_lambda": 0.9,  # Generalized Advantage Estimation (GAE) parameter. Balances bias vs. variance; lower values favor bias.
    "clip_range": 0.05,  # Clipping range for the PPO objective to prevent large policy updates. Keeps updates more conservative.
    "learning_rate": get_linear_fn(5e-4, 1e-6, total_timesteps*2),  # Linear decay from 5e-5 to 1e-6
}

# More detailed explanation:
#
# learning_rate: Controls how quickly or slowly the model updates its parameters. A very low value, like 1e-6, results in slow learning, which can sometimes prevent instability.
# n_steps: Determines how many steps of experience are collected before updating the policy. A larger n_steps provides more data for each update but requires more memory and computation.
# batch_size: The number of samples used to compute each gradient update. It affects the variance of the gradient estimate and the stability of learning.
# gamma: The discount factor, which defines how future rewards are weighted relative to immediate rewards. A high value (close to 1) makes the agent focus on long-term rewards.
# gae_lambda: A parameter used in the Generalized Advantage Estimation (GAE) method, which helps reduce variance in the advantage estimates. It controls the trade-off between bias and variance.
# clip_range: The range within which the policy is clipped to prevent overly large updates, ensuring more stable training.


calf_filter = calfq_filter()
calf_filter.sampling_time_init(dt)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if the --notrain flag is provided
if not args.notrain:

    PPO.calf_filter = calf_filter
    PPO.collect_rollouts = collect_rollouts
    PPO.learn = learn
    # Create the PPO model with the specified hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=ppo_hyperparams["learning_rate"],
        n_steps=ppo_hyperparams["n_steps"],
        batch_size=ppo_hyperparams["batch_size"],
        gamma=ppo_hyperparams["gamma"],
        gae_lambda=ppo_hyperparams["gae_lambda"],
        clip_range=ppo_hyperparams["clip_range"],
        verbose=1,
    )

    model.policy.to(device)

    model.calf_filter.init_policy(model.policy)
    # Create the plotting callback
    plotting_callback = PlottingCallback()

    # Train the model
    print("Training the model...")
    model.learn(total_timesteps=total_timesteps, callback=plotting_callback)
    # Save the model after training
    model.save("ppo_pendulum")
    # Close the plot after training
    plt.ioff()  # Turn off interactive mode
    # plt.show()  # Show the final plot
    # plt.close("all")   
else:
    print("Skipping training phase...")

# ====Evaluation: animated plot to show trained agent's performance

# Now enable rendering with pygame for testing
import pygame
env = gym.make("PendulumRenderFix-v0", render_mode="human")

# Load the model (if needed)
model = PPO.load("ppo_pendulum")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Переместите политику модели на нужное устройство
model.policy.to(device)
print(style.RED, device, style.RESET)

# Reset the environment
# obs, _ = env.reset()
obs, _ = env.reset(options={"angle": np.pi, "angular_velocity": 1.0})
cos_theta, sin_theta, angular_velocity = obs

# Initial critic value (expand dims to simulate batch input)
obs_tensor = torch.tensor(np.array([obs]), dtype=torch.float32).to(device)

last_good_value = model.policy.predict_values(obs_tensor)
last_good_model = model
last_good_action, _ = model.predict(obs)
print(last_good_action)
last_good_action = np.clip(last_good_action, -2.0, 2.0)

# Initialize total reward
total_reward = 0.0

# Run the simulation and render it
dt = 0.05  # Time step for the simulation

# Initialize pygame and set the display size
pygame.init()
# screen = pygame.display.set_mode((800, 600))  # Adjust the dimensions as needed

for step in range(500):
    # Compute the control action using the energy-based controller
    control_action = controller.compute(cos_theta, angular_velocity)
    control_action = np.clip([control_action], -2.0, 2.0)

    # Generate the action from the agent model
    agent_action, _ = model.predict(obs)
    agent_action = np.clip(agent_action, -2.0, 2.0)  # Clip to the valid range

    # Convert the current observation to a PyTorch tensor
    #! obs_tensor = torch.tensor([obs], dtype=torch.float32)
    obs_tensor = torch.tensor(np.array([obs]), dtype=torch.float32).to(device)

    current_value = model.policy.predict_values(obs_tensor)

    # action = calfq_filter.compute_action(agent_action,
    #                                      last_good_value,
    #                                      current_value
    #                                      )
###############################################################
    # # Evaluate the current state using the critic
    # current_value = model.policy.predict_values(obs_tensor)
    # # print("AIOSDASFASF", model.policy.state_dict("mlp_extractor.policy_net.2.weight"))
    # # print(model.policy) "mlp_extractor.policy_net.2.weight"
    # ppo_weights = model.policy.mlp_extractor.policy_net[2].weight
    # ppo_weights_numpy = ppo_weights.detach().cpu().numpy()
    # print(ppo_weights_numpy.size)
    # print(obs)
    # print(agent_action)
    # new_action = calf_filter.pass_or_replace(agent_action, np.array(obs), ppo_weights_numpy)
###############################################################

    print(style.YELLOW, "Curren value: {} vs LG value: {}".format(current_value, last_good_value), style.RESET)
    # Compare the critic values to decide which action to use
    if current_value > last_good_value:
        # Use the agent's action if the critic value has improved
        action = agent_action
        last_good_action = action 
        # Update the previous value for the next iteration
        last_good_value = current_value
    else:
        # Otherwise, fallback to the energy-based controller's action
        action = last_good_action
        # action = agent_action

    print(style.RED, action, style.RESET)

    # !DEBUG
    # action = control_action
    # !DEBUG

    # Step the environment using the selected action
    obs, reward, done, _, _ = env.step(action)
    env.render()

    # Update the observation
    cos_theta, sin_theta, angular_velocity = obs

    # Update the total reward
    total_reward += reward

    # Formatted print statement
    print(f"Step: {step + 1:3d} | Current Reward: {reward:7.2f} | Total Reward: {total_reward:10.2f}")

    # Wait for the next time step
    time.sleep(dt)

# Close the environment after the simulation
env.close()