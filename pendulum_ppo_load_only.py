import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from gymnasium.wrappers import TimeLimit
from mygym.my_pendulum import PendulumRenderFix
# Import the custom callback from callback.py
from callback.plotting_callback import PlottingCallback
from stable_baselines3.common.utils import get_linear_fn
from utilities.mlflow_logger import mlflow_monotoring, get_ml_logger

from wrapper.calf_wrapper import CALFWrapper, CALFEnergyPendulumWrapper, RelaxProb
from controller.energybased import EnergyBasedController
from wrapper.pendulum_wrapper import AddTruncatedFlagWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

import pandas as pd
import os
import numpy as np


os.makedirs("logs", exist_ok=True)

matplotlib.use("TkAgg")  # Try "Qt5Agg" if "TkAgg" doesn't work

# Register the environment
gym.envs.registration.register(
    id="PendulumRenderFix-v0",
    entry_point="mygym.my_pendulum:PendulumRenderFix",
)

def parse_args(configs):
        # Initialize the argument parser
    parser = argparse.ArgumentParser(description="PPO Training and Evaluation for Pendulum")
    parser.add_argument("--console", action="store_true", help="Disable graphical output for console-only mode")
    parser.add_argument("--log", action="store_true", help="Enable logging and printing of simulation data.")
    parser.add_argument("--notrain", 
                        action="store_true", 
                        help="Skip the training phase",
                        default=True)

    parser.add_argument("--loadstep", 
                        type=int,
                        help="Choose step to load checkpoint",
                        default=configs["total_timesteps"])

    parser.add_argument("--seed", 
                        type=int,
                        help="Choose random seed",
                        default=42)

    return parser.parse_args()


@mlflow_monotoring()
def main(**kwargs):
    # Use your custom environment for training
    env = gym.make("PendulumRenderFix-v0")
    if kwargs.get("use_mlflow"):
        loggers = get_ml_logger()
    env = TimeLimit(env, max_episode_steps=1000)  # Set a maximum number of steps per episode

    # Total number of agent-environment interaction steps for training
    total_timesteps = 500000

    args = parse_args(configs={
        "total_timesteps": total_timesteps
    })

    calf_hyperparams = {
        "calf_decay_rate": 0.01,
        "initial_relax_prob": 0.5,
        "relax_prob_base_step_factor": .95,
        "relax_prob_episode_factor": 0.
    }

    # ====Evaluation: animated plot to show trained agent's performance
    
    def make_env():
        def _init():
            env = PendulumRenderFix(render_mode="human" if not args.console else None)
            # env = PendulumRenderFix()
            # env = TimeLimit(env, max_episode_steps=1000)  # Set a maximum number of steps per episode
            env = CALFWrapper(
                env,
                fallback_policy=CALFEnergyPendulumWrapper(EnergyBasedController()),
                relax_decay=RelaxProb(calf_hyperparams["initial_relax_prob"], total_steps=1000),
                calf_decay_rate=calf_hyperparams["calf_decay_rate"],
                initial_relax_prob=calf_hyperparams["initial_relax_prob"],
                relax_prob_base_step_factor=calf_hyperparams["relax_prob_base_step_factor"],
                relax_prob_episode_factor=calf_hyperparams["relax_prob_episode_factor"],
                debug=False,
                logger=loggers
            )
                        
            return env
        return _init
    
    # Now enable rendering with pygame for testing
    import pygame
    
    env_agent = DummyVecEnv([make_env()])

    # Load the model (if needed)
    model = PPO.load(f"checkpoints/ppo_pendulum_{args.loadstep}_steps")
    if loggers:
        model.set_logger(loggers)


    # Reset the environments
    env_agent.env_method("copy_policy_model", model.policy)
    env_agent.seed(seed=args.seed)
    obs = env_agent.reset()
    
    info_dict = {
        "state": [],
        "action": [],
        "reward": [],
        "relax_probability": [],
        "calf_activated_count": [],
        "accumulated_reward": [],
    }
    accumulated_reward = 0
    n_step = 1000

    # Run the simulation with the trained agent
    for step_i in range(n_step):
        action, _ = model.predict(obs)

        # Dynamically handle four or five return values
        result = env_agent.step(action)  # Take a step in the environment
        if len(result) == 4:
            obs, reward, done, info = result
            truncated = False
        else:
            obs, reward, done, truncated, info = result

        accumulated_reward += reward

        info_dict["state"].append(obs[0])
        info_dict["action"].append(action[0])
        info_dict["reward"].append(reward)
        info_dict["relax_probability"].append(env_agent.get_attr("relax_prob").copy()[0])
        info_dict["calf_activated_count"].append(env_agent.get_attr("calf_activated_count").copy()[0])
        info_dict["accumulated_reward"].append(accumulated_reward.copy())
        model.logger.dump(step_i)
        if done:
            model.logger.dump(n_step)
            obs = env_agent.reset()  # Reset the agent's environment

    # Close the environments
    env_agent.close()

    df = pd.DataFrame(info_dict)
    file_name = f"pure_ppo_with_calfw_eval_{args.loadstep}_seed_{args.seed}.csv"

    if args.log:
        df.to_csv("logs/" + file_name)

    print("Case:", file_name)
    print(df.tail(2))

if __name__ == "__main__":
    main()
