# Visual PPO with CNNs

> [!IMPORTANT]  
> It's supposed that this subfolder is your working directory

## Idea
We want to train a PPO agent on the standard pendulum environment and its observation is a stack of RGB images.

## Aim
It leverages custom Convolutional Neural Networks (CNNs) to process image-based observations and enables effective policy learning with stacked frames.

### Train and Evaluate PPO (Pendulum Environment)
#### Training
To train an agent on the visual pendulum environment using stacked image frames:
    
```bash
python pendulum_visual_ppo.py
```

##### Checkpoints

The checkpoints have following formats:
```
# Checkpoints stored each 1000 steps
./artifact/checkpoints/ppo_pendulum_<step-number>_steps.zip 

# The best checkpoint after training
./artifact/checkpoints/ppo_pendulum.zip 
```

##### Configurations

The following configuration was used:

```python
total_timesteps = 131072
episode_timesteps = 1024
image_height = 64
image_width = 64
n_steps = 1024
parallel_envs = 8

# Hyperparameters for PPO
ppo_hyperparams = {
    "learning_rate": 4e-4,
    "n_steps": n_steps,
    "batch_size": 512,
    "gamma": 0.99,
    "gae_lambda": 0.9,
    "clip_range": 0.2,
}
```

... and 4 stacked frames.

#### Evaluation
After training, evaluate the agent with (example command):

```bash
python pendulum_visual_ppo.py --notrain
```

A successfully pre-trained agent can be found [here](../../artifacts/workable_visual_PPO4pendulum.zip).
It was run using the following command:

```bash
 python pendulum_visual_ppo.py --normalize
```

## Options

Option | Description |
| ----- |  ----- |
| `--notrain` | Skip training and only run evaluation |
| `--console` | Run in console-only mode (no graphical outputs) |
| `--normalize` | Normalize reward for stable learning |
| `--single-thread` | Use a single-threaded environment for training |

### Test a CNN for Visual Pendulum

To visualize and save the CNN feature maps for the visual pendulum:

```bash
python tests/test_visual_pendulum_cnn_stacked.py
```

#### Options for Testing CNN.

Option | Description |
| ----- |  ----- |
| `--model` | Path to a trained model (e.g., checkpoints/model.zip) |

This script simulates the environment and saves the extracted CNN feature maps, which can be used to evaluate the model's understanding of the environment.

#### Tested CNN Architecture

The custom CNN architecture used in this repository for visual PPO is as follows:

**Convolutional Layers**:

1. 32 filters, 8x8 kernel, stride 4
1. 64 filters, 4x4 kernel, stride 2
1. 64 filters, 3x3 kernel, stride 1

**Fully Connected Layer**:
256 neurons for feature extraction

**Output**:

Processes stacked input frames (e.g., 4 frames) for policy learning.
[Here](../../gfx/CNN_architecture_PPO4Pendulum.png) is a graphical representation of the CNN architecture.
