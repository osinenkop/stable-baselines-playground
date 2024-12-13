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
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecNormalize

from src.mygym.my_pendulum import PendulumVisual

from src.wrapper.pendulum_wrapper import ResizeObservation
from src.wrapper.pendulum_wrapper import AddTruncatedFlagWrapper
from src.wrapper.calf_wrapper import CALFNominalWrapper, CALFWrapper_CustomizedRelaxProb, RelaxProb

from src.utilities.intercept_termination import signal_handler
from src.utilities.mlflow_logger import mlflow_monotoring, get_ml_logger




class CALF_PPOPendulumWrapper(CALFNominalWrapper):
    def __init__(self, checkpoint_path, action_low, action_high, device="cuda"):
        self.model = PPO.load(checkpoint_path)
        self.device = device
        self.action_space_low = action_low
        self.action_space_high = action_high
    
    def compute_action(self, observation):
        with torch.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(observation, self.device)
            actions, _, _ = self.model.policy(obs_tensor)
            actions = actions.cpu().numpy()

        return np.clip(actions, self.action_space_low, self.action_space_high)

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
    "gamma": 0.98,  # Discount factor for future rewards. Closer to 1 means the agent places more emphasis on long-term rewards.
    "gae_lambda": 0.9,  # Generalized Advantage Estimation (GAE) parameter. Balances bias vs. variance; lower values favor bias.
    "clip_range": 0.01,  # Clipping range for the PPO objective to prevent large policy updates. Keeps updates more conservative.
    # "learning_rate": get_linear_fn(1e-4, 0.5e-5, total_timesteps),  # Linear decay from
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
        loggers = get_ml_logger()

    calf_hyperparams = {
        "calf_decay_rate": args.calf_decay_rate,
        "initial_relax_prob": args.calf_init_relax,
        "relax_prob_base_step_factor": .95,
        "relax_prob_episode_factor": 0.
    }

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
    env_agent = CALFWrapper_CustomizedRelaxProb(
                env_agent,
                relax_decay=RelaxProb(calf_hyperparams["initial_relax_prob"], total_steps=1000),
                fallback_policy=CALF_PPOPendulumWrapper(
                                    args.fallback_checkpoint,
                                    action_high=env_agent.action_space.high,
                                    action_low=env_agent.action_space.low
                                    ),
                calf_decay_rate=calf_hyperparams["calf_decay_rate"],
                initial_relax_prob=calf_hyperparams["initial_relax_prob"],
                relax_prob_base_step_factor=calf_hyperparams["relax_prob_base_step_factor"],
                relax_prob_episode_factor=calf_hyperparams["relax_prob_episode_factor"],
                debug=False,
                logger=loggers
            )

    # Load the normalization statistics if --normalsize is used
    if args.normalize:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--console", action="store_true", help="Disable graphical output for console-only mode")
    parser.add_argument("--normalize", action="store_true", help="Enable observation and reward normalization")
    parser.add_argument("--single-thread", action="store_true", help="Use DummyVecEnv for single-threaded environment")
    parser.add_argument("--loadstep", 
                        type=int,
                        help="Choose step to load checkpoint")
    parser.add_argument("--fallback-checkpoint", 
                        type=str,
                        help="Choose checkpoint to load for CALF fallback")
    parser.add_argument("--eval-checkpoint", 
                        type=str,
                        help="Choose checkpoint to load for base agent in evaluation")
    parser.add_argument("--log", action="store_true", help="Enable logging and printing of simulation data.")
    parser.add_argument("--seed", 
                        type=int,
                        help="Choose random seed",
                        default=42)
    parser.add_argument("--eval-name", 
                        type=str,
                        help="Choose experimental name for logging")
    parser.add_argument("--calf-init-relax", 
                        type=float,
                        help="Choose initial relax probability",
                        default=0.5)
    parser.add_argument("--calf-decay-rate", 
                        type=float,
                        help="Choose CALF decay rate",
                        default=0.01)
    args = parser.parse_args()

    main(args)
