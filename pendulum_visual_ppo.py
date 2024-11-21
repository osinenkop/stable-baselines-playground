import argparse
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from model.cnn import CustomCNN
from mygym.my_pendulum import PendulumVisual
from mygym.my_pendulum import NormalizeObservation
from callback.plotting_callback import PlottingCallback
from callback.grad_monitor_callback import GradientMonitorCallback
from callback.cnn_output_callback import SaveCNNOutputCallback
from stable_baselines3.common.utils import get_linear_fn
from gymnasium.wrappers import TimeLimit
from mygym.my_pendulum import ResizeObservation
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize

# Global parameters
total_timesteps = 131072 * 4
episode_timesteps = 256
image_height = 64
image_width = 64
save_model_every_steps = 8192 * 4
parallel_envs = 8

# Define the hyperparameters for PPO
ppo_hyperparams = {
    "learning_rate": 4e-4,  # The step size used to update the policy network. Lower values can make learning more stable.
    "n_steps": 512,  # Number of steps to collect before performing a policy update. Larger values may lead to more stable updates.
    "batch_size": 2048,  # Number of samples used in each update. Smaller values can lead to higher variance, while larger values stabilize learning.
    "gamma": 0.99,  # Discount factor for future rewards. Closer to 1 means the agent places more emphasis on long-term rewards.
    "gae_lambda": 0.9,  # Generalized Advantage Estimation (GAE) parameter. Balances bias vs. variance; lower values favor bias.
    "clip_range": 0.2,  # Clipping range for the PPO objective to prevent large policy updates. Keeps updates more conservative.
    # "learning_rate": get_linear_fn(1e-4, 0.5e-5, total_timesteps),  # Linear decay from
}

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--notrain", action="store_true", help="Skip training and only run evaluation")
    parser.add_argument("--console", action="store_true", help="Disable graphical output for console-only mode")
    parser.add_argument("--normalize", action="store_true", help="Enable observation and reward normalization")
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
            env = NormalizeObservation(env)  # Normalize observations
            env.reset(seed=seed)  # Set the seed using the new method
            return env
        return _init

    # Use SubprocVecEnv to run environments in parallel
    env = SubprocVecEnv([make_env(seed) for seed in range(parallel_envs)])

    # Apply reward and observation normalization if --normalize flag is provided
    if args.normalize:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)
        print("Reward normalization enabled. Observations are pre-normalized to [0, 1].")

    obs = env.reset()
    print("Environment reset successfully.")

    # Set random seed for reproducibility
    set_random_seed(42)

    # Define the policy_kwargs to use the custom CNN
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=32)
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

    # begin----Callbacks----

    # Predefine a fixed sample of observations (e.g., from a single environment reset)
    sample_env = PendulumVisual(render_mode="rgb_array")
    sample_env = ResizeObservation(sample_env, (image_height, image_width))
    sample_env = NormalizeObservation(sample_env)
    sample_obs, _ = sample_env.reset()

    # Ensure the sample is properly shaped for the CNN
    sample_obs = np.expand_dims(sample_obs, axis=0)  # Add batch dimension if needed

    # Set up the SaveCNNOutputCallback
    cnn_output_callback = SaveCNNOutputCallback(
        save_path="./cnn_outputs", 
        obs_sample=sample_obs, 
        every_n_steps=25
    )

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
    callback = CallbackList([checkpoint_callback, plotting_callback, gradient_monitor_callback, cnn_output_callback])

    # Example observation sample
    # obs_sample = np.random.rand(1, 3, image_height, image_width)  # Adjust dimensions as per your CNN input

    # # Instantiate the SaveCNNOutputCallback with the observation sample
    # print("DEBUG. Saving CNN features for a sample image")
    # save_cnn_output_callback = SaveCNNOutputCallback(
    #     save_path="./cnn_outputs",
    #     obs_sample=obs_sample,  # Provide the sample observation
    #     every_n_steps=250,  # Adjust as needed
    # )

    # end----Callbacks----

    # Train the model if --notrain flag is not provided
    if not args.notrain:
        print("Starting training ...")
        model.learn(total_timesteps=total_timesteps, callback=callback)
        model.save("ppo_visual_pendulum")

        # Save the normalization statistics if --normalize is used
        if args.normalize:
            env.save("vecnormalize_stats.pkl")

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
