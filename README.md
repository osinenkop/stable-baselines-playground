# Playground for stable-baselines and CALF

Description of CALF (Critic as Lyapunov Function) can be found in [this paper](https://arxiv.org/abs/2405.18118).

It's better to work in a virtual environment.
The instructions are similar to those in [regelum-playground](https://github.com/osinenkop/regelum-playground).

Don't forget to install `tkinter`.

Typical command to train and evaluate:

```shell
python pendulum_ppo.py
```

When training is done, just running the evaluation goes as:

```shell
python pendulum_ppocalf.py --notrain
```

## Test PPO with visual pendulum environment

```shell
python pendulum_visual_ppo.py
````

## Options

Option | Description |
| ----- |  ----- |
| `--notrain` | Skip training, just evaluate |
| `--console` | Only console output, no live learning curve plot |
| `--normalize` | Normalize observation |

## Test a CNN for visual pendulum environment

```shell
python -m test.test_visual_pendulum_simple
```
