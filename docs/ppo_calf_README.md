# PPO with CALF as a wrapper

## Running the Scripts

### Train and Evaluate PPO (Pendulum Environment + CALF Wrapper)

#### Training
To train a PPO agent on the pendulum environment + CALF Wrapper:

```shell
python pendulum_ppo_calf_dev.py
```

The following configuration was used:

```python
total_timesteps = 500000

# Hyperparameters for PPO
ppo_hyperparams = {
    "learning_rate": 5e-4, 
    "n_steps": 4000,
    "batch_size": 200,
    "gamma": 0.98,
    "gae_lambda": 0.9,
    "clip_range": 0.05,
    "learning_rate": get_linear_fn(5e-4, 1e-6, total_timesteps*2),
}

# Hyperparameters for CALF Wrapper
calf_hyperparams = {
    "calf_decay_rate": 0.005,
    "initial_relax_prob": 0.7,
    "relax_prob_base_step_factor": 0.99,
    "relax_prob_episode_factor": 0.0
}
```

#### Evaluation
After training, evaluate the agent with 
```shell
python pendulum_ppo_calf_dev.py --notrain
```

#### Options

Option | Description |
| ----- |  ----- |
| `--notrain` | Skip training and only run evaluation |


### Train and Evaluate vanilla PPO (Pendulum Environment)
To train a PPO agent on the normal pendulum environment:

```shell
python pendulum_ppo.py
```

The following configuration was used:

```python
total_timesteps = 500000

# Hyperparameters for PPO
ppo_hyperparams = {
    "learning_rate": 5e-4, 
    "n_steps": 4000,
    "batch_size": 200,
    "gamma": 0.98,
    "gae_lambda": 0.9,
    "clip_range": 0.05,
    "learning_rate": get_linear_fn(5e-4, 1e-6, total_timesteps*2),
}
```
#### Evaluation
After training, to evaluate a PPO agent on the normal pendulum environment:

```shell
python pendulum_ppo.py --notrain
```

Using the same training checkpouint, to evaluate a PPO agent on the pendulum environment + CALF Wrapper:

```shell
python pendulum_ppo_load_only.py --notrain
```

The following CALF configuration was used:
```python
env = CALFWrapper(
    env,
    fallback_policy=CALFEnergyPendulumWrapper(EnergyBasedController()),
    calf_decay_rate=0.01,
    initial_relax_prob=0.5,
    relax_prob_base_step_factor=0.95,
    relax_prob_episode_factor=0.,
    debug=False,
    logger=loggers
)
```

#### Options

Option | Description |
| ----- |  ----- |
| `--notrain` | Skip training and only run evaluation |
| `--seed` | Random seed to initialize initial state of the pendulum |
| `--loadstep` | Choose the checkpoint step want to load in the evaluation phase (i.e. 200000, 201000 500000) |
| `--console` | Run in console-only mode (no graphical outputs) |

#### Evaluation scripts
To evaluate vanilla PPO with and without CALF wrapper, use this pre-defined script:
```shell
source evaluation.sh
```
Or to run 30 seeds for each case with corresponding initial states:
```shell
source evaluation_loop.sh
```
