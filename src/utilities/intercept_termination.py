import signal
import sys
import os

# Define a global variable for the training loop
training = True

def save_model_and_data(model, rewards, gradients):
    """
    Custom function to save the model and any data before termination.
    """
    print("\nIntercepted termination signal! Saving model and data...")
    # Save the model
    model.save("ppo_visual_pendulum_terminated.zip")
    # Save rewards
    with open("episode_rewards_terminated.csv", "w") as f:
        for reward in rewards:
            f.write(f"{reward}\n")
    # Save gradients (example placeholder)
    with open("gradients_terminated.txt", "w") as f:
        for gradient in gradients:
            f.write(f"{gradient}\n")
    print("Model and data saved. Exiting gracefully.")

def signal_handler(signal_received, frame):
    """
    Signal handler for intercepting termination signals.
    """
    global training
    training = False  # Set training to false to exit the training loop safely
    print("\nTermination signal received. Preparing to stop...")
