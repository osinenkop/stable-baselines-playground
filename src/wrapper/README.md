# [`src/wrapper`](./): directory for the customized Environment Wrapper used in this project

This directory contains the Wrappers implementation of the project. Below is a detailed overview of each module and its purpose.

## Directory Structure

- [`calf_wrapper.py`](./calf_wrapper.py) - CALF related components
  - CALF Environment Wrapper and its variants
  - Modules used in the experiemnt [`ppo_pendulum_calf_wrapper_eval`](../../run/ppo_pendulum_calf_wrapper_eval):
    - **CALF_Wrapper**: This CALF wrapper filter use exponential Relax probability decay
    - **RelaxProbExponential**: Support exponential decay of Relax Probability
  - Modules used in the experiemnt [`ppo_vispendulum_self_boost`](../../run/ppo_vispendulum_self_boost):
    - **CALFWrapperSingleVecEnv**: This CALF wrapper filter, inherit from **CALF_Wrapper**, is designed to wrap a non-parallel vectorized environment.
    - **RelaxProbLinear**: Support linear decay of Relax Probability
- [`calf_fallback_wrapper.py`](./calf_fallback_wrapper.py) - A CALF Fallback Wrappers contain pre-defined fallback of CALFWrapper used for the following experiments:
  - **CALFEnergyPendulumWrapper** module: A outter layer for CALF fallback to get action from EnergyBasedController (used in [`ppo_pendulum_calf_wrapper_eval`](../../run/ppo_pendulum_calf_wrapper_eval))
  - **CALFPPOPendulumWrapper**(CALFNominalWrapper) module: A outter layer for CALF fallback to get action from a PPO checkpoint (used in [`ppo_vispendulum_self_boost`](../../run/ppo_vispendulum_self_boost)).
- [`pendulum_wrapper.py`](./pendulum_wrapper.py) - A customized Environment Wrappers
  - Standard Pendulum Environment Wrappers
  - Pendulum Environment Wrappers with visual observation

## Contributing Guidelines

### Code Modifications
- Add new core functionality to the [`src/wrapper/`](./) directory
- When modifying existing code, ensure backwards compatibility
- Verify that existing experiments and results remain valid
- Document all significant changes
