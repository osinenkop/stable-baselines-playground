# [`run/`](./): Training and evaluation scripts for all agents

This directory contains all experiment configurations, training scripts, and evaluation procedures. Each subdirectory represents a specific experiment or agent configuration.

---

## Current Experiments
### 1. [`ppo_vispendulum_default/`](./ppo_vispendulum_default)
- **Description**: PPO implementation for visual pendulum control with stacked frames.  

---

### 2. [`ppo_vispendulum_self_boost/`](./ppo_vispendulum_self_boost)
- **Description**: PPO implementation for visual pendulum control with stacked frames.  
  Includes evaluation with **CALFWrapper** using its trained checkpoints as both agent and fallback.

#### Main Modules:
- **`CALFWrapperSingleVecEnv`** (`src.wrapper.calf_wrapper`):  
  - A CALF wrapper for **non-parallel** vectorized environments with linear Relax Probability decay (`RelaxProbLinear`).  
- **`CALFPPOPendulumWrapper`** (`src.wrapper.calf_fallback_wrapper`):  
  - A layer enabling CALF fallback actions using a PPO checkpoint.  
- **`RelaxProbLinear`** (`src.wrapper.calf_wrapper`):  
  - Supports linear decay of Relax Probability.

---

### 3. [`ppo_pendulum_calf_wrapper_eval/`](./ppo_pendulum_calf_wrapper_eval)
- **Description**: PPO agent for the standard Pendulum environment, evaluated using **CALFWrapper**.  

#### Main Modules:
- **`CALF_Wrapper`** (`src.wrapper.calf_wrapper`):  
  - A CALF wrapper using exponential Relax Probability decay.  
- **`CALFEnergyPendulumWrapper`** (`src.wrapper.calf_fallback_wrapper`):  
  - A layer for CALF fallback actions using an Energy-Based Controller.
- **`RelaxProbExponential`** (`src.wrapper.calf_wrapper`):  
  - Supports exponential decay of Relax Probability.
---

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
