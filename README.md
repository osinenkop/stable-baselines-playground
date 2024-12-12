# Stable Baselines Playground with CALF and Visual PPO

![Visual PPO example on pendulum](./gfx/ppo_visual_pendulum.gif)

This repository is a playground for reinforcement learning using [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).
It extends classic environments like Gym's Pendulum with visual observations.

## Overview

### Visual PPO with CNNs
This repository also includes an extension of PPO (Proximal Policy Optimization) to visual environments. It leverages custom Convolutional Neural Networks (CNNs) to process image-based observations and enables effective policy learning with stacked frames.

### CALF (Critic as Lyapunov Function)
CALF provides a method to stabilize reinforcement learning using Lyapunov theory. For more details, refer to [this paper](https://arxiv.org/abs/2405.18118).

This feature is in development for this repository.

---

## Setting Up the Environment

To work with this repository, it is recommended to use a virtual environment. Refer to the instructions in the [regelum-playground](https://github.com/osinenkop/regelum-playground) for detailed steps.

Make sure to install the required dependencies, including `tkinter` for visualizations:

```bash
pip install -e .
```

**We recommend to use python-3.11.**

Some issues you may find their solution [here](docs/error_resolution.md).

## Running Experiments

All experiments are located in the [`run/`](./run) directory. To create a new experiment:
1. Create a new subfolder in the [`run/`](./run) directory with self-explanatory name
2. Include a detailed README.md file documenting the experiment
3. Follow the repository guidelines outlined below

For detailed information about specific agents, training procedures, and evaluation methods, refer to the README.md files within each subdirectory of [`run/`](./run).

## Repository Structure

The repository is organized into the following directories:

- [`src/`](./src) - Source code and core implementations
- [`run/`](./run) - Training and evaluation scripts
- [`snippets/`](./snippets) - Utility scripts and code examples
- [`analysis/`](./analysis) - Results analysis and visualization tools
- [`docs/`](./docs) - Project documentation (not used yet)
- [`gfx/`](./gfx) - Graphics and visual assets

## Contributing Guidelines

### Creating New Experiments
1. Create a new subdirectory in [`run/`](./run)
2. Include a comprehensive README.md that details:
   - Experiment setup and configuration
   - Launch instructions
   - Evaluation procedures
   - Results analysis methods
   - Artifact storage locations

### Code Modifications
- Add new core functionality to the [`src/`](./src) directory
- When modifying existing code, ensure backwards compatibility
- Verify that existing experiments and results remain valid
- Document all significant changes

### Analyzing Results

For experiment analysis:
1. Create a new subdirectory in [`analysis/`](./analysis) that matches your experiment's name in [`run/`](./run)
2. Use Jupyter notebooks or other analysis tools to document your findings
3. Include clear documentation of analysis methods and results

For cross-experiment analysis:
1. Create a new subdirectory in [`analysis/`](./analysis) with a descriptive name
2. Document the relationships between experiments being analyzed
3. Maintain clear references to the original experiment data

> Note: 
> All analysis code should be well-documented and reproducible.

## Author

Pavel Osinenko, 2024

