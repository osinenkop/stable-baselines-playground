
### Train visual PPO and Evaluate visual PPO (Pendulum Environment) with CALF Wrapper
#### Training
Now ppo trained with nomalized rewards and observation is utilized for this experiment:

```bash
python pendulum_visual_ppo.py --normalize
```

#### Evaluation
NOTICE: `--fallback-checkpoint` and `--eval-checkpoint` have to be defined in this step.
After training, evaluate the agent with (example command):

```bash
python pendulum_visual_ppo_load_only.py \
    --calf-init-relax 0.5 \
    --calf-decay-rate 0.01 \
    --fallback-checkpoint "backups/2024-12-09 101010/checkpoints/ppo_visual_pendulum_327680_steps.zip" \
    --eval-checkpoint "backups/2024-12-09 101010/checkpoints/ppo_visual_pendulum_655360_steps.zip" \
    --log --console --seed 42
```

#### Monitoring

Please have a look at MLFlow UI to explore more information about CALF variable during evaluation steps.
To run mlflow, use this command at the project directory:
```
mlflow ui
```

#### Options

Option | Description |
| ----- |  ----- |
| `--seed` | Random seed to initialize initial state of the pendulum |
| `--console` | Run in console-only mode (no graphical outputs) |
| `--log` | Enable logging and printing of simulation data |
| `--calf-init-relax` | Choose initial relax probability |
| `--calf-decay-rate` | Choose desired CALF decay rate |
| `--fallback-checkpoint` | Choose checkpoint to load for CALF fallback |
| `--eval-checkpoint` | Choose checkpoint to load for base agent in evaluation |
| `--eval-name` | Choose experimental logging name which will be stored in the "logs| folder |

Or to run 30 seeds with corresponding initial states, varying initial relax probabilities:
```
source evaluation_visual_loop.sh
```
