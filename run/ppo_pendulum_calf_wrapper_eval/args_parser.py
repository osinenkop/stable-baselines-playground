from dataclasses import dataclass
from stable_baselines3.common.utils import get_linear_fn

import tyro


@dataclass
class PPOHyperparameters:
    # The step size used to update the policy network. Lower values can make learning more stable.
    learning_rate: float = 5e-4

    # Number of steps to collect before performing a policy update. Larger values may lead to more stable updates.
    n_steps: int = 4000

    # Number of samples used in each update. Smaller values can lead to higher variance, while larger values stabilize learning.
    batch_size: int = 200

    # Discount factor for future rewards. Closer to 1 means the agent places more emphasis on long-term rewards.
    gamma: float = 0.98

    # Generalized Advantage Estimation (GAE) parameter. Balances bias vs. variance; lower values favor bias.
    gae_lambda: float = 0.9

    # Clipping range for the PPO objective to prevent large policy updates. Keeps updates more conservative.
    clip_range: float = 0.05

    # Whether to use generalized State Dependent Exploration (gSDE) instead of action noise exploration
    use_sde: bool = True

    # Sample a new noise matrix every n steps when using gSDE
    sde_sample_freq: int = 4

@dataclass
class ExperimentConfig:
    # PPO hyperparameters
    ppo: PPOHyperparameters

    # Skip training and only run evaluation
    notrain: bool = False

    # Disable graphical output for console-only mode
    console: bool = False

    # Choose step to load checkpoint
    # If loadstep is None, the best checkpoint would be loaded
    loadstep: int | None = None

    # Enable logging of simulation data.
    log: bool = False

    # Enable printing of simulation data.
    debug: bool = False

    # Choose random seed for environmental initial state
    seed: int = 42


@dataclass
class CALFEvalExperimentConfig(ExperimentConfig):
    ## TODO: Split CALF Parameters to a separate class

    # Choose checkpoint to load for CALF fallback
    calf_fallback_checkpoint: str | None = None

    # Choose initial relax probability
    calf_init_relax: float | None = 0.5
    
    # Choose CALF decay rate
    calf_decay_rate: float | None = 0.01

    # Choose a decay factor that relax_prob would decrease every time step
    # used
    relax_prob_base_step_factor: float = 0.95

    # Choose an episode decay factor that initial relax_prob at every episode
    # would decrease used
    relax_prob_episode_factor: float = 0.


def parse_args(config, overide_default=None):
    return tyro.cli(config,
                    default=overide_default)

if __name__ == "__main__":
    parse_args(ExperimentConfig, 
                overide_default=ExperimentConfig(
                    ppo=PPOHyperparameters()
                ))
