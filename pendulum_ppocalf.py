import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
# import gym
import argparse
import numpy as np
import time
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit
#! from gym.wrappers import TimeLimit

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

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3 import A2C

from copy import deepcopy

from gymnasium import spaces
#! from gym import spaces

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

import math

from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import Wrapper
#! from gym import Wrapper


SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


# def make_env(env_id, seed):
#     def _init():
#         env = gym.make(env_id)
#         env.seed(seed)  # Устанавливаем seed для среды
#         return env
#     return _init

# class CustomVecFrameStack(VecFrameStack):
#     def reset(self, **kwargs):
#         kwargs.pop("seed", None)  # Удаляем аргумент seed, если он есть
#         return super().reset(**kwargs)
    
# class SingleReturnResetWrapper(Wrapper):
#     def reset(self, **kwargs):
#         obs, _ = super().reset(**kwargs)
#         return obs  # Возвращаем только наблюдения

# class CustomDummyVecEnv(DummyVecEnv):
#     def reset(self, **kwargs):
#         # Удаляем аргумент "seed", если он передан
#         kwargs.pop("seed", None)
#         return super().reset(**kwargs)

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
        self.calf_filter_delay = 5
        print("CALFQ Filter init")

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

    def value_reset(self): ...
        # self.best_value_local = None
    
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
    
    # def compute_action(self, action, observation, last_good_Q_value, Q_value, current_policy, obs_tensor):
    #     # print(last_good_Q_value)
    #     if ( Q_value - last_good_Q_value ) >= (self.nu):

    #         self.last_good_Q_value = Q_value
    #         self.best_policy = deepcopy(current_policy)
    #         new_action, _, _ = self.best_policy(obs_tensor)
    #         return new_action
    #     if (np.random.random()<=self.replacing_probability):
    #         return action
        
    #     action, _, _ = self.best_policy(obs_tensor)
    #     return action
    
def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        iteration: int,
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
        print("Iteration", iteration)
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        self.policy = calf_filter.get_last_good_model()
        print(style.YELLOW, self.policy.mlp_extractor.policy_net[2].weight, style.RESET)
        calf_filter.value_reset()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)
            # print(n_steps)

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
                                                               self.calf_filter.best_policy_global.predict_values(obs_tensor), 
                                                               self.policy.predict_values(obs_tensor),
                                                               self.policy,
                                                               obs_tensor,
                                                               iteration)
            
            if isinstance(clipped_actions, torch.Tensor):
                clipped_actions = clipped_actions.cpu().detach().numpy()
            else:
                clipped_actions = np.array(clipped_actions)

            clipped_actions = np.clip(clipped_actions, self.action_space.low, self.action_space.high)

            # TODO Update CaLF foreach episode
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            if dones[-1]:
                self.calf_filter.update_global_policy()
                # if self.calf_filter.policy_update is True:
                #     self.calf_filter.update(self.policy)
                # else:
                #     self.calf_filter.policy_update = True
                #     print("Policy Updated")

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

        
        print(style.BLUE, calf_filter.get_last_good_model().mlp_extractor.policy_net[2].weight, style.RESET)

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
            # print(self.num_timesteps)
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps, iteration=iteration)

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


# environment_name = "ATARI"
environment_name = "PENDULUM"

# Initialize the argument parser
parser = argparse.ArgumentParser(description="PPO Training and Evaluation for Pendulum")
parser.add_argument("--notrain", action="store_true", help="Skip the training phase")
parser.add_argument("--atari", action="store_true", help="Run atari env")
parser.add_argument("--pendulum", action="store_true", help="Run pendulum env")


# Parse the arguments
args = parser.parse_args()
matplotlib.use("TkAgg")  # Try "Qt5Agg" if "TkAgg" doesn't work
# Register the environment
gym.envs.registration.register(
    id="PendulumRenderFix-v0",
    entry_point="mygym.my_pendulum:PendulumRenderFix",
)
# Use your custom environment for training
dt = 0.05  # Action time step for the simulation
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

calf_filter = calfq_filter()
calf_filter.sampling_time_init(dt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if environment_name == "ATARI":

    env = make_atari_env("Pong-v4", n_envs=4) # ATARI
    # env = VecFrameStack(env, n_stack=4) # ATARI
    # env = SingleReturnResetWrapper(env)
    # env = CustomVecFrameStack(env, n_stack=4) # ATARI
    # env = CustomDummyVecEnv([lambda: env])

    env = TimeLimit(env, max_episode_steps=1000)  # Set a maximum number of steps per episode

if environment_name == "PENDULUM":
    env = gym.make("PendulumRenderFix-v0") # PENDULUM ENV
    env = TimeLimit(env, max_episode_steps=1000)  # Set a maximum number of steps per episode
    '''
    More detailed explanation:

    learning_rate: Controls how quickly or slowly the model updates its parameters. A very low value, like 1e-6, results in slow learning, which can sometimes prevent instability.
    n_steps: Determines how many steps of experience are collected before updating the policy. A larger n_steps provides more data for each update but requires more memory and computation.
    batch_size: The number of samples used to compute each gradient update. It affects the variance of the gradient estimate and the stability of learning.
    gamma: The discount factor, which defines how future rewards are weighted relative to immediate rewards. A high value (close to 1) makes the agent focus on long-term rewards.
    gae_lambda: A parameter used in the Generalized Advantage Estimation (GAE) method, which helps reduce variance in the advantage estimates. It controls the trade-off between bias and variance.
    clip_range: The range within which the policy is clipped to prevent overly large updates, ensuring more stable training.
    '''


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
    model.save("ppo_pendulum_modi_2")
    # Close the plot after training
    plt.ioff()  # Turn off interactive mode
else:
    print("Skipping training phase...")

# ====Evaluation: animated plot to show trained agent's performance
# Now enable rendering with pygame for testing
import pygame
env = gym.make("PendulumRenderFix-v0", render_mode="human")
# Load the model (if needed)
model = PPO.load("ppo_pendulum_modi_2")
model.policy.to(device)
print(style.RED, device, style.RESET)

# Reset the environment
obs, _ = env.reset(options={"angle": np.pi, "angular_velocity": 1.0})
# cos_theta, sin_theta, angular_velocity = obs

# Initial critic value (expand dims to simulate batch input)
obs_tensor = torch.tensor(np.array([obs]), dtype=torch.float32).to(device)

# Initialize total reward
total_reward = 0.0

# Initialize pygame and set the display size
pygame.init()
# screen = pygame.display.set_mode((800, 600))  # Adjust the dimensions as needed

for step in range(500):

    # Generate the action from the agent model
    agent_action, _ = model.predict(obs)
    #! agent_action = np.clip(agent_action, -2.0, 2.0)  # Clip to the valid range PENDULUM

    # Convert the current observation to a PyTorch tensor
    #! obs_tensor = torch.tensor([obs], dtype=torch.float32)
    obs_tensor = torch.tensor(np.array([obs]), dtype=torch.float32).to(device)

    current_value = model.policy.predict_values(obs_tensor)

    print(style.RED, agent_action, style.RESET)

    # Step the environment using the selected action
    obs, reward, done, _, _ = env.step(agent_action)
    #! env.render() # PENDULUM  
    env.render("human")

    # Update the observation
    # cos_theta, sin_theta, angular_velocity = obs

    # Update the total reward
    total_reward += reward

    # Formatted print statement
    print(f"Step: {step + 1:3d} | Current Reward: {reward:7.2f} | Total Reward: {total_reward:10.2f}")

    # Wait for the next time step
    time.sleep(dt)

# Close the environment after the simulation
env.close()
