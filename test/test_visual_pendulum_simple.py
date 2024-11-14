import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from mygym.my_pendulum import PendulumVisual

env = PendulumVisual()
env.reset()
for _ in range(100):
    env.render()
