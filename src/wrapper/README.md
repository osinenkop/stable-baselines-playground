# [`src/wrapper`](./): directory for the customized Environment Wrapper used in this project

This directory contains the Wrappers implementation of the project. Below is a detailed overview of each module and its purpose.

## Directory Structure



### [`calf_wrapper.py`](./calf_wrapper.py) 
Contains CALF-related components, including wrappers and utilities for managing relax probability decay.

- **Modules for Experiment [`ppo_pendulum_calf_wrapper_eval`](../../run/ppo_pendulum_calf_wrapper_eval):**
  - **CALF_Wrapper:** A CALF wrapper filter that uses exponential relax probability decay.
  - **RelaxProbExponential:** Supports exponential decay of the relax probability.

- **Modules for Experiment [`ppo_vispendulum_self_boost`](../../run/ppo_vispendulum_self_boost):**
  - **CALFWrapperSingleVecEnv:** Inherits from **CALF_Wrapper**, designed for non-parallel vectorized environments.
  - **RelaxProbLinear:** Supports linear decay of relax probability.

---

### [`calf_fallback_wrapper.py`](./calf_fallback_wrapper.py)
Implements CALF fallback wrappers with predefined fallback strategies for CALFWrapper. These are tailored for specific experiments:

- **CALFEnergyPendulumWrapper:**
  - Serves as an outer layer for CALF fallback, sourcing actions from the **EnergyBasedController**.
  - Used in experiment [`ppo_pendulum_calf_wrapper_eval`](../../run/ppo_pendulum_calf_wrapper_eval).

- **CALFPPOPendulumWrapper (CALFNominalWrapper):**
  - Serves as an outer layer for CALF fallback, sourcing actions from a PPO checkpoint.
  - Used in experiment [`ppo_vispendulum_self_boost`](../../run/ppo_vispendulum_self_boost).

---
### [`pendulum_wrapper.py`](./pendulum_wrapper.py)
Contains customized wrappers for the standard and visual observation pendulum environments:
- Standard Pendulum Environment Wrappers
- Pendulum Environment Wrappers with visual observation

## Contributing Guidelines

### Code Modifications
- Add new core functionality to the [`src/wrapper/`](./) directory
- When modifying existing code, ensure backwards compatibility
- Verify that existing experiments and results remain valid
- Document all significant changes
