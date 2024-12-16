import argparse


def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--notrain", action="store_true", help="Skip training and only run evaluation")
    parser.add_argument("--console", action="store_true", help="Disable graphical output for console-only mode")
    parser.add_argument("--normalize", action="store_true", help="Enable observation and reward normalization", 
                        default=True)
    parser.add_argument("--eval-normalize", action="store_true", help="Enable observation and reward normalization for evaluation")
    parser.add_argument("--single-thread", action="store_true", help="Use DummyVecEnv for single-threaded environment")
    parser.add_argument("--loadstep", 
                        type=int,
                        help="Choose step to load checkpoint")
    parser.add_argument("--log", action="store_true", help="Enable logging and printing of simulation data.")
    parser.add_argument("--debug", action="store_true", help="Enable printing of simulation data.")
    parser.add_argument("--seed", 
                        type=int,
                        help="Choose random seed",
                        default=42)
    parser.add_argument("--eval-checkpoint", 
                        type=str,
                        help="Choose step to load checkpoint")
    parser.add_argument("--eval-name", 
                        type=str,
                        help="Choose experimental name for logging")
    args = parser.parse_args()

    return args