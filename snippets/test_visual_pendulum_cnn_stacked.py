import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed

from src.mygym.my_pendulum import PendulumVisual
from src.mygym.my_pendulum import PendulumVisualNoArrowParallelizable

from src.wrapper.pendulum_wrapper import ResizeObservation
from src.callback.cnn_output_callback import SaveCNNOutputCallback
from src.utilities.clean_cnn_outputs import clean_cnn_outputs

from src.model.cnn import CustomCNN

from src.agent.debug_ppo import DebugPPO


# Global parameters
total_timesteps = 10
image_height = 64
image_width = 64
num_frames = 4 # Adjust to match the paramter in the frame stacking wrapper of the environment!

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test CNN features visualization on PendulumVisual environment.")
    parser.add_argument("--model", type=str, default=None, help="Path to the trained PPO model zip file")
    args = parser.parse_args()

    # Clean CNN outputs folder
    clean_cnn_outputs("./test/cnn_outputs_test")

    # Create the environment
    env = DummyVecEnv([
        lambda: ResizeObservation(PendulumVisualNoArrowParallelizable(render_mode="rgb_array"),
        (image_height, image_width))
        ])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    # Load or initialize the model
    if args.model:
        print(f"Loading model from {args.model}")
        model = PPO.load(
            args.model,
            custom_objects={
                "features_extractor_class": CustomCNN,
                "features_extractor_kwargs": dict(features_dim=256, num_frames=num_frames),
            }
        )
    else:
        print("Initializing a random model")

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256, num_frames=num_frames)
        )

        # policy_kwargs = dict(features_extractor_kwargs=dict(features_dim=256, num_frames=4))
        model = DebugPPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0)

    # Test CNN features visualization
    print("Testing CNN with stacked frames and callback...")
    test_callback = SaveCNNOutputCallback(
        save_path="./test/cnn_outputs_test",
        every_n_steps=1,
        max_channels=3
    )

    # Make sure state attribute is initialized in env
    obs = env.reset()

    for step in range(total_timesteps):
        print(f"Simulation step {step} done")
        # action = env.action_space.sample()  # Generate a random action
        action, _ = model.predict(obs) # Get action from the model
        obs, reward, _, _ = env.step(action)

        angular_velocity = env.envs[0].state[1]  # Assuming the environment state includes angular velocity
        time_step_ms = env.envs[0].dt * 1000

        obs_tensor = torch.tensor(obs[0], dtype=torch.float32).unsqueeze(0).to(model.device)

        with torch.no_grad():
            cnn_features = model.policy.features_extractor.get_layer_features(obs_tensor)

        # Save the visualizations
        test_callback._save_frame_visualization(
            obs=obs[0],
            features=cnn_features,
            step=step,
            reward=float(reward),
            action=float(action[0]),
            angular_velocity=angular_velocity,
            time_step_ms=time_step_ms
        )

    print("Finished visualizing frames and CNN features.")

if __name__ == "__main__":
    main()
