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
from stable_baselines3.common.vec_env import VecNormalize

# Global parameters
total_timesteps=131072*12
episode_timesteps=4096
image_height=32
image_width=32
save_model_every_steps=8192*4
parallel_envs=8

# Define the hyperparameters for PPO
ppo_hyperparams = {
    "learning_rate": 5e-4,  # The step size used to update the policy network. Lower values can make learning more stable.
    "n_steps": 4096,  # Number of steps to collect before performing a policy update. Larger values may lead to more stable updates.
    "batch_size": 4096 * parallel_envs,  # Number of samples used in each update. Smaller values can lead to higher variance, while larger values stabilize learning.
    "gamma": 0.99,  # Discount factor for future rewards. Closer to 1 means the agent places more emphasis on long-term rewards.
    "gae_lambda": 0.9,  # Generalized Advantage Estimation (GAE) parameter. Balances bias vs. variance; lower values favor bias.
    "clip_range": 0.05,  # Clipping range for the PPO objective to prevent large policy updates. Keeps updates more conservative.
    "learning_rate": get_linear_fn(5e-4, 1e-6, total_timesteps*2),  # Linear decay from 5e-5 to 1e-6
}

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--notrain", action="store_true", help="Skip training and only run evaluation")
    parser.add_argument("--console", action="store_true", help="Disable graphical output for console-only mode")
    args = parser.parse_args()

    # Check if the --console flag is used
    if args.console:
        import matplotlib
        matplotlib.use('Agg')  # Use a non-GUI backend to disable graphical output

    # Function to create the environment and set the seed
    def make_env(seed):
        def _init():
            env = PendulumVisual()
            env = TimeLimit(env, max_episode_steps=episode_timesteps)  # Set a maximum number of steps per episode
            env = ResizeObservation(env, (image_height, image_width))  # Resize the observation
            env.reset(seed=seed)  # Set the seed using the new method
            return env
        return _init

    # Use SubprocVecEnv to run environments in parallel
    env = SubprocVecEnv([make_env(seed) for seed in range(parallel_envs)])

    # Set up a checkpoint callback to save the model every 'save_freq' steps
    checkpoint_callback = CheckpointCallback(
        save_freq=save_model_every_steps,  # Save the model periodically
        save_path="./checkpoints",  # Directory to save the model
        name_prefix="ppo_visual_pendulum"
    )

    obs = env.reset()
    print("Environment reset successfully.")

    # Set random seed for reproducibility
    set_random_seed(42)

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

    # Instantiate a plotting call back to show live learning curve
    plotting_callback = PlottingCallback()

    # If --console flag is set, disable the plot and just save the data
    if args.console:
        plotting_callback.figure = None  # Disable plotting
        print("Console mode: Graphical output disabled. Episode rewards will be saved to 'episode_rewards.csv'.")

    # Combine both callbacks using CallbackList
    callback = CallbackList([checkpoint_callback, plotting_callback])

    # Train the model if --notrain flag is not provided
    if not args.notrain:
        print("Starting training ...")
        model.learn(total_timesteps=total_timesteps, callback=callback)
        # model.learn(total_timesteps=total_timesteps)
        model.save("ppo_visual_pendulum")
        print("Training completed.")
    else:
        print("Skipping training. Loading the saved model...")
        model = PPO.load("ppo_visual_pendulum")

    # Visual evaluation after training or loading
    print("Starting evaluation...")

    # Environment for the agent (using 'rgb_array' mode)
    env_agent = PendulumVisual(render_mode="rgb_array")
    env_agent = ResizeObservation(env_agent, (image_height, image_width))  # Resize for the agent

    # Environment for visualization (using 'human' mode)
    env_display = PendulumVisual(render_mode="human")

    # Reset the environments
    obs, _ = env_agent.reset()
    env_display.reset()

    # Run the simulation with the trained agent
    for _ in range(3000):
        action, _ = model.predict(obs)
        # action = env_agent.action_space.sample()  # Generate a random action
        obs, reward, done, _, _ = env_agent.step(action)  # Take a step in the environment

        env_display.step(action)  # Step in the display environment to show animation

        if done:
            obs, _ = env_agent.reset()  # Reset the agent's environment
            env_display.reset()  # Reset the display environment

    # Close the environments
    env_agent.close()
    env_display.close()