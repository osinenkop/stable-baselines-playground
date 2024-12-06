import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import signal

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.preprocessing import is_image_space

from model.cnn import CustomCNN

from mygym.my_pendulum import PendulumVisual
from mygym.my_pendulum import PendulumVisualNoArrowParallelizable

from wrapper.pendulum_wrapper import NormalizeObservation
from wrapper.pendulum_wrapper import ResizeObservation
from wrapper.pendulum_wrapper import LoggingWrapper
from wrapper.pendulum_wrapper import AddTruncatedFlagWrapper

from callback.plotting_callback import PlottingCallback
from callback.grad_monitor_callback import GradientMonitorCallback
from callback.cnn_output_callback import SaveCNNOutputCallback

from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers.frame_stack import FrameStack

from agent.debug_ppo import DebugPPO

from utilities.clean_cnn_outputs import clean_cnn_outputs
from utilities.intercept_termination import save_model_and_data, signal_handler
from utilities.mlflow_logger import mlflow_monotoring, get_ml_logger

import pandas as pd
import os


os.makedirs("logs", exist_ok=True)

# Global parameters
total_timesteps = 131072 * 10
episode_timesteps = 1024
image_height = 64
image_width = 64
save_model_every_steps = 8192 / 4
n_steps = 1024
parallel_envs = 4

# Define the hyperparameters for PPO
ppo_hyperparams = {
    "learning_rate": 4e-4,  # The step size used to update the policy network. Lower values can make learning more stable.
    "n_steps": n_steps,  # Number of steps to collect before performing a policy update. Larger values may lead to more stable updates.
    "batch_size": 512,  # Number of samples used in each update. Smaller values can lead to higher variance, while larger values stabilize learning.
    "gamma": 0.99,  # Discount factor for future rewards. Closer to 1 means the agent places more emphasis on long-term rewards.
    "gae_lambda": 0.9,  # Generalized Advantage Estimation (GAE) parameter. Balances bias vs. variance; lower values favor bias.
    "clip_range": 0.2,  # Clipping range for the PPO objective to prevent large policy updates. Keeps updates more conservative.
    # "learning_rate": get_linear_fn(1e-4, 0.5e-5, total_timesteps),  # Linear decay from
}

# Global variables for graceful termination
is_training = True
episode_rewards = []  # Collect rewards during training
gradients = []  # Placeholder for gradients during training

@mlflow_monotoring()
def main(**kwargs):
    # Register signal handlers
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame))

    if kwargs.get("use_mlflow"):
        loggers = get_ml_logger()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--notrain", action="store_true", help="Skip training and only run evaluation")
    parser.add_argument("--console", action="store_true", help="Disable graphical output for console-only mode")
    parser.add_argument("--normalize", action="store_true", help="Enable observation and reward normalization")
    parser.add_argument("--single-thread", action="store_true", help="Use DummyVecEnv for single-threaded environment")
    parser.add_argument("--loadstep", 
                        type=int,
                        help="Choose step to load checkpoint")
    parser.add_argument("--log", action="store_true", help="Enable logging and printing of simulation data.")
    parser.add_argument("--seed", 
                        type=int,
                        help="Choose random seed",
                        default=42)
    args = parser.parse_args()

    # Check if the --console flag is used
    if args.console:
        matplotlib.use('Agg')  # Use a non-GUI backend to disable graphical output
    else:
        matplotlib.use("TkAgg")

    # Train the model if --notrain flag is not provided
    if not args.notrain:

        # Define a global variable for the training loop
        is_training = True

        # Function to create the base environment
        def make_env(seed):
            def _init():
                env = PendulumVisualNoArrowParallelizable()
                # env = LoggingWrapper(env)  # For debugging: log each step. Comment out by default
                env = TimeLimit(env, max_episode_steps=episode_timesteps)
                env = ResizeObservation(env, (image_height, image_width))
                env.reset(seed=seed)
                return env
            return _init

        # Environment setup based on --single-thread flag
        if args.single_thread:
            print("Using single-threaded environment (DummyVecEnv).")
            env = DummyVecEnv([make_env(0)])
        else:
            print("Using multi-threaded environment (SubprocVecEnv).")
            env = SubprocVecEnv([make_env(seed) for seed in range(parallel_envs)])

        # Apply VecFrameStack to stack frames along the channel dimension
        env = VecFrameStack(env, n_stack=4)

        # Apply VecTransposeImage
        env = VecTransposeImage(env)

        # Apply reward and observation normalization if --normalize flag is provided
        if args.normalize:
            env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)
            print("Reward normalization enabled. Observations are pre-normalized to [0, 1].")

        env.seed(seed=args.seed)
        obs = env.reset()
        print("Environment reset successfully.")

        # Set random seed for reproducibility
        set_random_seed(args.seed)

        # Define the policy_kwargs to use the custom CNN
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256, num_frames=4)  # Adjust num_frames as needed
        )

        # Create the PPO agent using the custom feature extractor
        model = DebugPPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=ppo_hyperparams["learning_rate"],
            n_steps=ppo_hyperparams["n_steps"],
            batch_size=ppo_hyperparams["batch_size"],
            gamma=ppo_hyperparams["gamma"],
            gae_lambda=ppo_hyperparams["gae_lambda"],
            clip_range=ppo_hyperparams["clip_range"],
            verbose=1,
            use_sde=True,
            sde_sample_freq=10,
        )
        
        if kwargs.get("use_mlflow"):    
            model.set_logger(loggers)

        print("Model initialized successfully.")

        # Set up a checkpoint callback to save the model every 'save_freq' steps
        checkpoint_callback = CheckpointCallback(
            save_freq=save_model_every_steps,  # Save the model periodically
            save_path="./checkpoints",  # Directory to save the model
            name_prefix="ppo_visual_pendulum"
        )

        # Instantiate a plotting callback to show the live learning curve
        plotting_callback = PlottingCallback()

        # Instantiate the GradientMonitorCallback
        gradient_monitor_callback = GradientMonitorCallback()    

        # If --console flag is set, disable the plot and just save the data
        if args.console:
            plotting_callback.figure = None  # Disable plotting
            print("Console mode: Graphical output disabled. Episode rewards will be saved to 'episode_rewards.csv'.")

        # Combine both callbacks using CallbackList
        callback = CallbackList([
            checkpoint_callback,
            plotting_callback,
            gradient_monitor_callback
            ])

        print("Starting training ...")

        try:
            model.learn(total_timesteps=total_timesteps, callback=callback)
        except KeyboardInterrupt:
            print("Training interrupted. Saving model and data...")
            save_model_and_data(model, episode_rewards, gradients)
        finally:
            print("Training completed or interrupted.")

        model.save("ppo_visual_pendulum")

        # Save the normalization statistics if --normalize is used
        if args.normalize:
            env.save("vecnormalize_stats.pkl")

        env.close()
        print("Training completed.")
    else:
        print("Skipping training. Loading the saved model...")
        model = PPO.load("ppo_visual_pendulum")

        # Load the normalization statistics if --normalize is used
        if args.normalize:
            env = VecNormalize.load("vecnormalize_stats.pkl", env)
            env.training = False  # Set to evaluation mode
            env.norm_reward = False  # Disable reward normalization for evaluation

    # Visual evaluation after training or loading
    print("Starting evaluation...")

    # Environment for the agent (using 'rgb_array' mode)
    env_agent = DummyVecEnv([
        lambda: AddTruncatedFlagWrapper(
            ResizeObservation(PendulumVisual(render_mode="rgb_array"), (image_height, image_width))
        )
    ])
    env_agent = VecFrameStack(env_agent, n_stack=4)
    env_agent = VecTransposeImage(env_agent)

    # Environment for visualization (using 'human' mode)
    env_display = PendulumVisual(render_mode="rgb_array" if args.console else "human")

    # Reset the environments
    env_agent.seed(seed=args.seed)
    
    obs = env_agent.reset()
    env_display.reset(seed=args.seed)
    
    info_dict = {
        "state": [],
        "action": [],
        "reward": [],
        "accumulated_reward": [],
    }
    accumulated_reward = 0

    # Run the simulation with the trained agent
    for _ in range(1000):
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
        env_display.step(action)  # Step in the display environment to show animation

        if done:
            obs = env_agent.reset()  # Reset the agent's environment
            env_display.reset()  # Reset the display environment

        accumulated_reward += reward

        info_dict["state"].append(obs)
        info_dict["action"].append(action)
        info_dict["reward"].append(reward)
        info_dict["accumulated_reward"].append(accumulated_reward.copy())

    # Close the environments
    env_agent.close()
    env_display.close()

    df = pd.DataFrame(info_dict)
    file_name = f"visual_ppo_eval_{args.loadstep}_seed_{args.seed}.csv"

    if args.log:
        df.to_csv("logs/" + file_name)

    print("Case:", file_name)
    print(df.tail(2))

if __name__ == "__main__":
    main()    