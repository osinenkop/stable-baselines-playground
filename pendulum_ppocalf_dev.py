import argparse
import signal
import matplotlib

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.evaluation import evaluate_policy

from model.cnn import CustomCNN

from agent.ppo_calf import PPO_CALF

from mygym.my_pendulum import PendulumRenderFix

from wrapper.pendulum_wrapper import NormalizeObservation
from wrapper.pendulum_wrapper import LoggingWrapper
from wrapper.pendulum_wrapper import AddTruncatedFlagWrapper

from callback.plotting_callback import PlottingCallback
from callback.grad_monitor_callback import GradientMonitorCallback
from callback.cnn_output_callback import SaveCNNOutputCallback

from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers.frame_stack import FrameStack

from agent.debug_ppo import DebugPPO

from utilities.mlflow_logger import mlflow_monotoring, get_ml_logger
from utilities.clean_cnn_outputs import clean_cnn_outputs
from utilities.intercept_termination import save_model_and_data, signal_handler

from wrapper.calf_wrapper import CALFWrapper, CALFEnergyPendulumWrapper
from controller.energybased import EnergyBasedController

# Global parameters
total_timesteps = 500000
episode_timesteps = 1024
image_height = 64
image_width = 64
save_model_every_steps = 8192 / 4
n_steps = 1024
parallel_envs = 1


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

calf_hyperparams = {
    "calf_decay_rate": 0.001,
    "initial_relax_prob": 0.4,
    "relax_prob_base_step_factor": 0.9,
    "relax_prob_episode_factor": 0.01
}

# Global variables for graceful termination
is_training = True
episode_rewards = []  # Collect rewards during training
gradients = []  # Placeholder for gradients during training


@mlflow_monotoring
def main(**kwarg):
    # Register signal handlers
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame))
    
    hyperparams = kwarg.get("hyperparams")
    if kwarg.get("use_mlflow"):
        loggers = get_ml_logger()

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--notrain", action="store_true", help="Skip training and only run evaluation")
    parser.add_argument("--console", action="store_true", 
                        help="Disable graphical output for console-only mode")
    parser.add_argument("--normalize", action="store_true", 
                        help="Enable observation and reward normalization")
    parser.add_argument("--single-thread", action="store_true", default=True,
                        help="Use DummyVecEnv for single-threaded environment")
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
                env = PendulumRenderFix()
                env = TimeLimit(env, max_episode_steps=1000)  # Set a maximum number of steps per episode
                env = CALFWrapper(env, 
                                  fallback_policy=CALFEnergyPendulumWrapper(EnergyBasedController()),
                                  calf_decay_rate=hyperparams["calf_decay_rate"],
                                  initial_relax_prob=hyperparams["initial_relax_prob"],
                                  relax_prob_base_step_factor=hyperparams["relax_prob_base_step_factor"],
                                  relax_prob_episode_factor=hyperparams["relax_prob_episode_factor"],
                                  logger=loggers
                )
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

        # Apply reward and observation normalization if --normalize flag is provided
        if args.normalize:
            env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)
            print("Reward normalization enabled. Observations are pre-normalized to [0, 1].")

        obs = env.reset()
        print("Environment reset successfully.")

        # Set random seed for reproducibility
        set_random_seed(42)

        # Create the PPO agent using the custom feature extractor
        model = PPO_CALF(
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

        model.set_logger(loggers)
        
        print("Model initialized successfully.")        

        # Set up a checkpoint callback to save the model every 'save_freq' steps
        # checkpoint_callback = CheckpointCallback(
        #     save_freq=save_model_every_steps,  # Save the model periodically
        #     save_path="./checkpoints",  # Directory to save the model
        #     name_prefix="ppo_calf_pendulum"
        # )

        # Instantiate a plotting callback to show the live learning curve
        plotting_callback = PlottingCallback()

        # Instantiate the GradientMonitorCallback
        # gradient_monitor_callback = GradientMonitorCallback()

        # If --console flag is set, disable the plot and just save the data
        if args.console:
            plotting_callback.figure = None  # Disable plotting
            print("Console mode: Graphical output disabled. Episode rewards will be saved to 'episode_rewards.csv'.")

        # Combine both callbacks using CallbackList
        callback = CallbackList([
            # checkpoint_callback,
            plotting_callback,
            # gradient_monitor_callback
            ])

        print("Starting training ...")

        try:
            model.learn(total_timesteps=total_timesteps, callback=callback)
        except KeyboardInterrupt:
            print("Training interrupted. Saving model and data...")
            save_model_and_data(model, episode_rewards, gradients)
        finally:
            print("Training completed or interrupted.")

        model.save("ppo_calf_pendulum")

        # Save the normalization statistics if --normalize is used
        if args.normalize:
            env.save("vecnormalize_stats.pkl")

        print("Training completed.")

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
        print("Policy Eval")
        print(f"mean_reward: {mean_reward}, std_reward: {std_reward}")
    else:
        print("Skipping training. Loading the saved model...")
        model = PPO_CALF.load("ppo_calf_pendulum")

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
            CALFWrapper(
                PendulumRenderFix(), 
                fallback_policy=CALFEnergyPendulumWrapper(EnergyBasedController()),
                calf_decay_rate=calf_hyperparams["calf_decay_rate"],
                initial_relax_prob=calf_hyperparams["initial_relax_prob"],
                relax_prob_base_step_factor=calf_hyperparams["relax_prob_base_step_factor"],
                relax_prob_episode_factor=calf_hyperparams["relax_prob_episode_factor"],
            )
        )
    ])

    # Environment for visualization (using 'human' mode)
    env_display = PendulumRenderFix(render_mode="human")

    # Reset the environments
    obs = env_agent.reset()
    env_display.reset()

    # Run the simulation with the trained agent
    # for _ in range(3000):
    for _ in range(100):
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

    # Close the environments
    env_agent.close()
    env_display.close()

if __name__ == "__main__":
    main(hyperparams=calf_hyperparams)    