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

from wrapper.calf_wrapper import CALFWrapper, CALFEnergyPendulumWrapper
from controller.energybased import EnergyBasedController
from wrapper.pendulum_wrapper import AddTruncatedFlagWrapper
from stable_baselines3.common.vec_env import DummyVecEnv


# Initialize the argument parser
parser = argparse.ArgumentParser(description="PPO Training and Evaluation for Pendulum")
parser.add_argument("--notrain", 
                    action="store_true", 
                    help="Skip the training phase",
                    default=True)

# Parse the arguments
args = parser.parse_args()

matplotlib.use("TkAgg")  # Try "Qt5Agg" if "TkAgg" doesn't work

# Register the environment
gym.envs.registration.register(
    id="PendulumRenderFix-v0",
    entry_point="mygym.my_pendulum:PendulumRenderFix",
)

@mlflow_monotoring
def main(**kwargs):
    # Use your custom environment for training
    env = gym.make("PendulumRenderFix-v0")
    if kwargs.get("use_mlflow"):
        loggers = get_ml_logger()
    env = TimeLimit(env, max_episode_steps=1000)  # Set a maximum number of steps per episode

    # Total number of agent-environment interaction steps for training
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

    

    # Check if the --notrain flag is provided
    if not args.notrain:

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
        if kwargs.get("use_mlflow"):    
            model.set_logger(loggers)

        # Create the plotting callback
        plotting_callback = PlottingCallback()

        checkpoint_callback = CheckpointCallback(
            save_freq=1000,  # Save the model periodically
            save_path="./checkpoints",  # Directory to save the model
            name_prefix="ppo_pendulum"
            )

        # Combine both callbacks using CallbackList
        callback = CallbackList([
            checkpoint_callback,
            plotting_callback,
            # gradient_monitor_callback
            ])

        # Train the model
        print("Training the model...")
        model.learn(total_timesteps=total_timesteps, callback=callback)
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
    env_agent = DummyVecEnv([
        lambda: AddTruncatedFlagWrapper(
            CALFWrapper(
                PendulumRenderFix(render_mode="human"), 
                fallback_policy=CALFEnergyPendulumWrapper(EnergyBasedController()),
                calf_decay_rate=0.01,
                initial_relax_prob=0,
                relax_prob_base_step_factor=0.95,
                relax_prob_episode_factor=0.,
                debug=True
            )
        )
    ])

    # Load the model (if needed)
    model = PPO.load("checkpoints/ppo_pendulum_200000_steps")

    # Reset the environments
    obs = env_agent.reset()
    env_agent.env_method("copy_policy_model", model.policy)

    # Run the simulation with the trained agent
    # for _ in range(3000):
    for _ in range(1000):
        action, _ = model.predict(obs)

        # Dynamically handle four or five return values
        result = env_agent.step(action)  # Take a step in the environment
        if len(result) == 4:
            obs, reward, done, info = result
            truncated = False
        else:
            obs, reward, done, truncated, info = result

        if done:
            obs = env_agent.reset()  # Reset the agent's environment

    # Close the environments
    env_agent.close()


if __name__ == "__main__":
    main()
