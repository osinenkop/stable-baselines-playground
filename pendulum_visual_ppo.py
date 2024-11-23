import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

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

from wrapper.pendulum_wrapper import NormalizeObservation
from wrapper.pendulum_wrapper import ResizeObservation

from callback.plotting_callback import PlottingCallback
from callback.grad_monitor_callback import GradientMonitorCallback
from callback.cnn_output_callback import SaveCNNOutputCallback

from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers.frame_stack import FrameStack

from agent.debug_ppo import DebugPPO

from utilities.clean_cnn_outputs import clean_cnn_outputs

# Global parameters
total_timesteps = 131072 * 4
episode_timesteps = 256
image_height = 64
image_width = 64
save_model_every_steps = 8192 * 4
n_steps = 256
parallel_envs = 8

# Define the hyperparameters for PPO
ppo_hyperparams = {
    "learning_rate": 4e-3,  # The step size used to update the policy network. Lower values can make learning more stable.
    "n_steps": n_steps,  # Number of steps to collect before performing a policy update. Larger values may lead to more stable updates.
    "batch_size": 256,  # Number of samples used in each update. Smaller values can lead to higher variance, while larger values stabilize learning.
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

    # Call the function to clean the CNN outputs folder
    clean_cnn_outputs("./cnn_outputs")

    # Check if the --console flag is used
    if args.console:
        import matplotlib
        matplotlib.use('Agg')  # Use a non-GUI backend to disable graphical output

    # Function to create the base environment
    def make_env(seed):
        def _init():
            env = PendulumVisual()
            env = TimeLimit(env, max_episode_steps=episode_timesteps)
            env = ResizeObservation(env, (image_height, image_width))
            env.reset(seed=seed)
            return env
        return _init

    # Create SubprocVecEnv
    # env = SubprocVecEnv([make_env(seed) for seed in range(parallel_envs)])
    env = DummyVecEnv([make_env(0)])  # Single-threaded environment

    # Apply VecFrameStack to stack frames along the channel dimension
    env = VecFrameStack(env, n_stack=4)

    # Debug: Check the final observation space
    # print(f"Observation space before VecTransposeImage: {env.observation_space}")

    # Apply VecTransposeImage
    env = VecTransposeImage(env)

    # Debug: Final check
    # print(f"Final env observation space: {env.observation_space}")
    # input("Press Enter to continue...")

    # Debug: sample observation
    obs = env.reset()
    print(f"Sample observation shape: {obs.shape}")

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
    )
    print("Model initialized successfully.")

    # HERE A SHORT TEST OF CNN==============================================
    from callback.cnn_output_callback import SaveCNNOutputCallback

    print("Testing CNN with stacked frames and callback...")

    # Create the callback instance for testing
    test_callback = SaveCNNOutputCallback(
        save_path="./cnn_outputs_test",
        every_n_steps=1,  # For testing, save at every step
        max_channels=3    # Visualize up to 3 channels per layer
    )

    # Generate multiple frames by interacting with the environment
    num_steps = 10
    stacked_obs = []

    for i in range(num_steps):
        print(f"We are at step {i}")

        # Generate a random action
        random_action = env.action_space.sample()
        print(f"Random action: {random_action}")

        # Take a step in the environment
        obs, reward, _, info = env.step(random_action)

        # Get angular velocity and time step
        angular_velocity = env.envs[0].state[1]  # Assuming the environment state includes angular velocity
        time_step_ms = env.envs[0].dt * 1000    # Convert time step to milliseconds

        # Debug observation shape
        print(f"Step {i} observation shape: {obs.shape}")

        # Pass through the CNN and save visualizations
        stacked_obs.append(obs[0])  # Collect the first environment's observation
        obs_tensor = torch.tensor(obs[0], dtype=torch.float32).unsqueeze(0).to(model.device)

        with torch.no_grad():
            cnn_features = model.policy.features_extractor.get_layer_features(obs_tensor)

        # Save the visualization
        test_callback._save_frame_visualization(
            obs=obs[0],
            features=cnn_features,
            step=i,
            reward=float(reward),  # Convert to scalar
            action=float(random_action[0]),  # Extract the scalar value from the array
            angular_velocity=angular_velocity,
            time_step_ms=time_step_ms
        )

    print("Finished visualizing frames and CNN features.")

    input("Press Enter to continue...")

    # Set up the SaveCNNOutputCallback
    cnn_output_callback = SaveCNNOutputCallback(
        save_path="./cnn_outputs",
        every_n_steps=n_steps,  # Save every so many training steps
        max_channels=3      # Visualize up to 3 channels per layer
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
    callback = CallbackList([
        checkpoint_callback,
        plotting_callback,
        gradient_monitor_callback,
        cnn_output_callback
        ])

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
