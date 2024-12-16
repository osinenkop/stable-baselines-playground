import argparse


def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="PPO Training and Evaluation for Pendulum")
    parser.add_argument("--console", action="store_true", help="Disable graphical output for console-only mode")
    parser.add_argument("--log", action="store_true", help="Enable logging and printing of simulation data.")
    parser.add_argument("--debug", action="store_true", help="Enable printing of simulation data.")
    parser.add_argument("--notrain", 
                        action="store_true", 
                        help="Skip the training phase")
    parser.add_argument("--loadstep", 
                        type=int,
                        help="Choose step to load checkpoint")
    parser.add_argument("--seed", 
                        type=int,
                        help="Choose random seed",
                        default=42)
    # Parse the arguments
    args = parser.parse_args()

    return args