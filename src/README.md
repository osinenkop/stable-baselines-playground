# [`src/`](./): directory for the source code for the project

This directory contains the core implementation of the project. Below is a detailed overview of each subdirectory and its purpose.

## Directory Structure

- [`agent/`](./agent) - Agent Components
  - Houses custom agent implementations
  - Includes reinforcement learning agents and their variants
- [`callback/`](./callback) - Training Infrastructure
  - Contains callback implementations for Stable-Baselines3 agents
  - Handles training events, logging, and model checkpointing
- [`controller/`](./controller) - Control Algorithms
  - Implements deterministic control algorithms
  - Features PD controllers and energy-based control systems
  - No training required for these classical control approaches
- [`model/`](./model) - Neural Network Architectures
  - Contains custom neural network architectures
  - Specifically designed for visual-based agents
  - Includes CNN implementations and policy networks
- [`mygym/`](./mygym) - Custom Environment Implementations
  - Custom OpenAI Gym environment implementations
  - Defines reward functions, state spaces, and dynamics
- [`utilities/`](./utilities) - Utility Functions and Classes
  - File I/O operations
  - Data visualization
  - MLflow experiment tracking
  - Performance metrics
- [`wrapper/`](./wrapper) - Environment Wrappers
  - Environment modification layers
  - Features the **CALFWrapper** (Critic As Lyapunov Function)
  - Implements state/reward transformations and additional monitoring
