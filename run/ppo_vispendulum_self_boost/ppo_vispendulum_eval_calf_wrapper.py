import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import signal
import time
import pandas as pd
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecNormalize

from src.mygym.my_pendulum import PendulumVisual

from src.wrapper.pendulum_wrapper import ResizeObservation
from src.wrapper.pendulum_wrapper import AddTruncatedFlagWrapper
from src.wrapper.calf_wrapper import CALFWrapperSingleVecEnv, RelaxProbLinear
from src.wrapper.calf_fallback_wrapper import CALFPPOPendulumWrapper

from src.utilities.intercept_termination import signal_handler
from src.utilities.mlflow_logger import mlflow_monotoring, get_ml_logger

from run.ppo_vispendulum_self_boost.args_parser import parse_args, CALFEvalExperimentConfig, ARG_REQUIRED, PPOHyperparameters


os.makedirs("logs", exist_ok=True)

# Global parameters
image_height = 64
image_width = 64

# Default setting, can be overwritten by arguments
calf_hyperparams = {
    "calf_decay_rate": 0.01,
    "initial_relax_prob": 0.5,
    "relax_prob_base_step_factor": 0, # Unused inside wrapper
    "relax_prob_episode_factor": 0, # Unused inside wrapper
}

# Global variables for graceful termination
is_training = True
episode_rewards = []  # Collect rewards during training
gradients = []  # Placeholder for gradients during training

# @rerun_if_error
@mlflow_monotoring(subfix="_0.1")
def main(args, **kwargs):
    # Register signal handlers
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame))

    if kwargs.get("use_mlflow"):
        loggers = get_ml_logger(args.debug)

    # Check if the --console flag is used
    if args.console:
        matplotlib.use('Agg')  # Use a non-GUI backend to disable graphical output
    else:
        matplotlib.use("TkAgg")

    print("Skipping training. Loading the saved model...")
    if args.eval_checkpoint:
        model = PPO.load(args.eval_checkpoint)
    elif args.loadstep:
        model = PPO.load(f"./artifacts/checkpoints/ppo_visual_pendulum_{args.loadstep}_steps")
    else:
        model = PPO.load("./artifacts/ppo_visual_pendulum")

    model.set_logger(loggers)

    # Visual evaluation after training or loading
    print("Starting evaluation...")
    
    # Now enable rendering with pygame for testing
    import pygame
    
    # Environment for the agent (using 'rgb_array' mode)
    env_agent = DummyVecEnv([
        lambda: AddTruncatedFlagWrapper(
            ResizeObservation(PendulumVisual(render_mode="rgb_array"), 
                              (image_height, image_width))
        )
    ])

    env_agent = VecFrameStack(env_agent, n_stack=4)
    env_agent = VecTransposeImage(env_agent)
    env_agent = CALFWrapperSingleVecEnv(
                env_agent,
                relax_decay=RelaxProbLinear(
                    calf_hyperparams["initial_relax_prob"], 
                    total_steps=1000),
                fallback_policy=CALFPPOPendulumWrapper(
                                    args.calf_fallback_checkpoint,
                                    action_high=env_agent.action_space.high,
                                    action_low=env_agent.action_space.low
                                    ),
                calf_decay_rate=calf_hyperparams["calf_decay_rate"],
                debug=False,
                logger=loggers
            )

    # Load the normalization statistics if --normalize is used
    if args.eval_normalize:
        env_agent = VecNormalize.load("./artifacts/checkpoints/vecnormalize_stats.pkl", env_agent)
        env_agent.training = False  # Set to evaluation mode
        env_agent.norm_reward = False  # Disable reward normalization for evaluation

    # env_agent.env_method("copy_policy_model", model.policy)
    env_agent.copy_policy_model(model.policy)
    
    # Reset the environments
    env_agent.seed(seed=args.seed)
    
    obs, _ = env_agent.reset()
    # obs = env_agent.reset()
    
    info_dict = {
        "state": [],
        "action": [],
        "reward": [],
        "relax_probability": [],
        "calf_activated_count": [],
        "accumulated_reward": [],
    }
    accumulated_reward = np.float32(0)
    fig, ax = plt.subplots()

    # Run the simulation with the trained agent again run until truncated
    for step_i in range(1000):
        action, _ = model.predict(obs)
        # action = env_agent.action_space.sample()  # Generate a random action

        # Dynamically handle four or five return values
        result = env_agent.step(action)  # Take a step in the environment
        if len(result) == 4:
            obs, reward, done, info = result
            truncated = False
        else:
            obs, reward, done, truncated, info = result

        # Handle the display environment
        # env_display.step(action)  # Step in the display environment to show animation
        if done:
            obs = env_agent.reset()  # Reset the agent's environment

        accumulated_reward += reward

        info_dict["state"].append(obs[0])
        info_dict["action"].append(action[0])
        info_dict["reward"].append(reward)
        info_dict["relax_probability"].append(env_agent.relax_prob)
        info_dict["calf_activated_count"].append(env_agent.calf_activated_count)
        info_dict["accumulated_reward"].append(accumulated_reward.copy())
        model.logger.dump(step_i)
        
        if not args.console:
            ax.imshow(obs[0][-3:].transpose((1, 2, 0)).copy())
            ax.axis("off")
            plt.pause(1/30)
            
    # Close the environments
    env_agent.close()

    print("Get outside of the evaluation")

    df = pd.DataFrame(info_dict)

    if args.eval_name:
        file_name = f"ppo_vispendulum_eval_calf_{args.eval_name}_seed_{args.seed}.csv"
    else:
        file_name = f"ppo_vispendulum_eval_calf_{args.loadstep}_seed_{args.seed}.csv"

    if args.log:
        df.to_csv("logs/" + file_name)

    print("Case:", file_name)
    print(df.drop(columns=["state"]).tail(2))

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args(CALFEvalExperimentConfig, 
                      overide_default=CALFEvalExperimentConfig(
                          calf_init_relax=calf_hyperparams["initial_relax_prob"],
                          calf_decay_rate=calf_hyperparams["calf_decay_rate"],
                          calf_fallback_checkpoint=ARG_REQUIRED,
                          ppo=PPOHyperparameters()
                      ))

    main(args)
