import argparse
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from model.cnn import CustomCNN
from mygym.my_pendulum import PendulumVisual
from callback.plotting_callback import PlottingCallback  # Import your existing callback
from stable_baselines3.common.utils import get_linear_fn
from gymnasium.wrappers import TimeLimit
from mygym.my_pendulum import ResizeObservation 
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

# Global parameters
total_timesteps=100000
episode_timesteps=1000
image_height=32
image_width=32
save_model_every_steps=10000

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--notrain", action="store_true", help="Skip training and only run evaluation")
    args = parser.parse_args()

    # Function to create the environment and set the seed
    def make_env(seed):
        def _init():
            env = PendulumVisual()
            env = TimeLimit(env, max_episode_steps=episode_timesteps)  # Set a maximum number of steps per episode
            env = ResizeObservation(env, (image_height, image_width))  # Resize the observation
            env.reset(seed=seed)  # Set the seed using the new method
            return env
        return _init

    # Create a DummyVecEnv with 4 parallel environments
    env = DummyVecEnv([make_env(seed) for seed in range(4)])

    # Use SubprocVecEnv to run environments in parallel
    env = SubprocVecEnv([make_env(seed) for seed in range(4)])

    # env = VecTransposeImage(env)  # Transpose image observations for PyTorch

    # Set a maximum number of steps per episode
    # env = TimeLimit(env, max_episode_steps=episode_timesteps)

    # Set up a checkpoint callback to save the model every 'save_freq' steps
    checkpoint_callback = CheckpointCallback(
        save_freq=save_model_every_steps,  # Save the model periodically
        save_path="./checkpoints",  # Directory to save the model
        name_prefix="ppo_visual_pendulum"
    )

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

    # DEBUG
    obs = env.reset()
    print("Environment reset successfully.")

    # Set random seed for reproducibility
    set_random_seed(42)
    # env.seed(42)  # Use the updated seed method

    # Define the policy_kwargs to use the custom CNN
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256)
    )

    # Create the PPO agent using the custom feature extractor
    model = PPO(
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
    )
    print("Model initialized successfully.")

    # Total number of agent-environment interaction steps for training
    total_timesteps = 500000

    # Instantiate a plotting call back to show live learning curve
    plotting_callback = PlottingCallback()

    # Combine both callbacks using CallbackList
    callback = CallbackList([checkpoint_callback, plotting_callback])

    # Train the model if --notrain flag is not provided
    if not args.notrain:
        print("Starting training ...")
        model.learn(total_timesteps=total_timesteps, callback=plotting_callback)
        # model.learn(total_timesteps=total_timesteps)
        model.save("ppo_visual_pendulum")
        print("Training completed.")
    else:
        print("Skipping training. Loading the saved model...")
        model = PPO.load("ppo_visual_pendulum")

    # Visual evaluation after training or loading
    print("Starting visual evaluation...")

    # Visual evaluation after training or loading
    env = DummyVecEnv([lambda: PendulumVisual()])
    env = VecTransposeImage(env)
    obs = env.reset()

    # Reload the environment for evaluation
    env = PendulumVisual()
    obs, _ = env.reset()

    # Run the simulation with the trained agent
    for _ in range(500):
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)
        env.render()
        if done:
            obs, _ = env.reset()

    # Close the environment
    env.close()
