# [`run/`](./): Training and evaluation scripts for all agents

This directory contains all experiment configurations, training scripts, and evaluation procedures. Each subdirectory represents a specific experiment or agent configuration.


## Current Experiments

- [`ppo_vispendulum_default/`](./ppo_vispendulum_default)
  - PPO implementation for visual pendulum control with stacked frames
  
- [`ppo_vispendulum_self_boost/`](./ppo_vispendulum_self_boost)
  - TODO: add description

- [`ppo_pendulum_calf_wrapper_eval/`](./ppo_pendulum_calf_wrapper_eval)
  - PPO agent on standard pendulum
  - evaluation with CALFWrapper

## Contributing Guidelines
### Creating New Experiments
1. Create a new subdirectory in [`run/`](./) with self-explanatory name
2. Include a comprehensive README.md that details:
   - Experiment setup and configuration
   - Launch instructions
   - Evaluation procedures
   - Results analysis methods
   - Artifact storage locations


## Related Directories

- [`../analysis/`](../analysis) - For result analysis and visualization
- [`../src/`](../src) - Core implementation and utilities