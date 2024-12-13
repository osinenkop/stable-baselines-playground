# [`run/`](./): Training and evaluation scripts for all agents

This directory contains all experiment configurations, training scripts, and evaluation procedures. Each subdirectory represents a specific experiment or agent configuration.


## Current Experiments

- [`ppo_vispendulum_default/`](./ppo_vispendulum_default)
  - PPO implementation for visual pendulum control with stacked frames
  
- [`ppo_vispendulum_self_boost/`](./ppo_vispendulum_self_boost)
  - PPO implementation for visual pendulum control with stacked frames
  - Evaluation with CALFWrapper using its trained checkpoints as an agent and CALF fallback.
  - Main related modules:
    - src.wrapper.calf_wrapper.**CALFWrapper_CustomizedRelaxProb**: This CALF wrapper filter use `RelaxProb` decay
    - **CALF_PPOPendulumWrapper**(CALFNominalWrapper): A firm layer for CALF fallback to get action from a checkpoint of PPO (defined in [`Python script`](../../run/ppo_vispendulum_self_boost/ppo_vispendulum_eval_calf_wrapper.py))
    - src.wrapper.calf_wrapper.**RelaxProb**: Support linear decay of Relax Probability


- [`ppo_pendulum_calf_wrapper_eval/`](./ppo_pendulum_calf_wrapper_eval)
  - PPO agent on standard pendulum
  - evaluation with CALFWrapper
  - Main CALF related modules:
    - src.wrapper.calf_wrapper.**CALF_Wrapper**: This CALF wrapper filter use exponential Relax probability decay
    - src.wrapper.calf_wrapper.**CALFEnergyPendulumWrapper**: A firm layer for CALF fallback to get action from EnergyBasedController

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
