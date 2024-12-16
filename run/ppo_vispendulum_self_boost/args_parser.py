from dataclasses import dataclass

import tyro


ARG_REQUIRED = tyro.MISSING

@dataclass
class PPOHyperparameters:
    # The step size used to update the policy network. Lower values can make learning more stable.
    learning_rate: float = 4e-4

    # Number of steps to collect before performing a policy update. Larger values may lead to more stable updates.
    n_steps: int = 1024

    # Number of samples used in each update. Smaller values can lead to higher variance, while larger values stabilize learning.
    batch_size: int = 512

    # Discount factor for future rewards. Closer to 1 means the agent places more emphasis on long-term rewards.
    gamma: float = 0.99

    # Generalized Advantage Estimation (GAE) parameter. Balances bias vs. variance; lower values favor bias.
    gae_lambda: float = 0.9

    # Clipping range for the PPO objective to prevent large policy updates. Keeps updates more conservative.
    clip_range: float = 0.2


@dataclass
class ExperimentConfig:
    # PPO hyperparameters
    ppo: PPOHyperparameters

    # Skip training and only run evaluation
    notrain: bool = False

    # Disable graphical output for console-only mode
    console: bool = False

    # Enable observation and reward normalization
    normalize: bool = True

    # Enable observation and reward normalization for evaluation
    eval_normalize: bool = False

    # Use DummyVecEnv for single-threaded environment
    single_thread: bool = False

    # Choose step to load checkpoint
    loadstep: int | None = None

    # Enable logging of simulation data.
    log: bool = False

    # Enable printing of simulation data.
    debug: bool = False

    # Choose random seed for environmental initial state
    seed: int = 42

    # Choose step to load checkpoint
    # If eval_checkpoint is None, the best checkpoint would be loaded
    eval_checkpoint: str | None = None

    # Choose experimental name for logging
    eval_name: str = "default"

@dataclass
class CALFEvalExperimentConfig(ExperimentConfig):
    ## TODO: Split CALF Parameters to a separate class

    # Choose checkpoint to load for CALF fallback
    calf_fallback_checkpoint: str | None = None

    # Choose initial relax probability
    calf_init_relax: float | None = 0.5
    
    # Choose CALF decay rate
    calf_decay_rate: float | None = 0.01


def parse_args(config, overide_default=None):
    return tyro.cli(config,
                    default=overide_default)


if __name__ == "__main__":
    parse_args(ExperimentConfig, 
                overide_default=ExperimentConfig(
                    ppo=PPOHyperparameters(
                        learning_rate=4e-4,
                        n_steps=2, 
                        batch_size=512, 
                        gamma=0.98,
                        gae_lambda=0.9,
                        clip_range=0.01,
                    )
                ))
