import matplotlib
import gymnasium as gym
import argparse

from stable_baselines3 import PPO

from gymnasium.wrappers import TimeLimit
from src.mygym.my_pendulum import PendulumRenderFix
from src.utilities.mlflow_logger import mlflow_monotoring, get_ml_logger
from src.wrapper.calf_wrapper import CALFWrapper
from src.wrapper.calf_fallback_wrapper import CALFEnergyPendulumWrapper
from src.controller.energybased import EnergyBasedController

from stable_baselines3.common.vec_env import DummyVecEnv

import pandas as pd
import os

from run.ppo_pendulum_calf_wrapper_eval.args_parser import parse_args, CALFEvalExperimentConfig, PPOHyperparameters


os.makedirs("logs", exist_ok=True)

matplotlib.use("TkAgg")  # Try "Qt5Agg" if "TkAgg" doesn't work

# Register the environment
gym.envs.registration.register(
    id="PendulumRenderFix-v0",
    entry_point="src.mygym.my_pendulum:PendulumRenderFix",
)

# global environment (Default setting, can be overwritten by arguments)
calf_hyperparams = {
    "calf_decay_rate": 0.01,
    "initial_relax_prob": 0.5,
    "relax_prob_base_step_factor": .95,
    "relax_prob_episode_factor": 0.
}

@mlflow_monotoring()
def main(args, **kwargs):
    # Use your custom environment for training
    env = gym.make("PendulumRenderFix-v0")
    if kwargs.get("use_mlflow"):
        loggers = get_ml_logger(args.debug)
    env = TimeLimit(env, max_episode_steps=1000)  # Set a maximum number of steps per episode

    # ====Evaluation: animated plot to show trained agent's performance
    
    def make_env():
        def _init():
            env = PendulumRenderFix(render_mode="human" if not args.console else None)
            # env = PendulumRenderFix()
            # env = TimeLimit(env, max_episode_steps=1000)  # Set a maximum number of steps per episode
            env = CALFWrapper(
                env,
                fallback_policy=CALFEnergyPendulumWrapper(EnergyBasedController()),
                calf_decay_rate=args.calf_decay_rate,
                initial_relax_prob=args.calf_init_relax,
                relax_prob_base_step_factor=args.relax_prob_base_step_factor,
                relax_prob_episode_factor=args.relax_prob_episode_factor,
                debug=False,
                logger=loggers
            )
                        
            return env
        return _init
    
    # Now enable rendering with pygame for testing
    import pygame
    
    env_agent = DummyVecEnv([make_env()])

    # Load the model (if needed)
    if args.loadstep:
        model = PPO.load(f"artifacts/checkpoints/ppo_pendulum_{args.loadstep}_steps")
    else:
        model = PPO.load(f"artifacts/checkpoints/ppo_pendulum")

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
    args = parse_args(CALFEvalExperimentConfig, 
                    overide_default=CALFEvalExperimentConfig(
                        calf_init_relax=calf_hyperparams["initial_relax_prob"],
                        calf_decay_rate=calf_hyperparams["calf_decay_rate"],
                        relax_prob_base_step_factor=calf_hyperparams["relax_prob_base_step_factor"],
                        relax_prob_episode_factor=calf_hyperparams["relax_prob_episode_factor"],
                        ppo=PPOHyperparameters()
                    ))
    main(args)
