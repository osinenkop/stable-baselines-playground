import gymnasium as gym
import numpy as np
from src.mygym.my_pendulum import PendulumVisual
from src.mygym.my_pendulum import ResizeObservation 

# Environment for the agent (using 'rgb_array' mode)
env_agent = PendulumVisual(render_mode="rgb_array")
env_agent = ResizeObservation(env_agent, (32, 32))  # Resize for the agent

# Environment for visualization (using 'human' mode)
env_display = PendulumVisual(render_mode="human")

# Reset the environments
obs, _ = env_agent.reset()
env_display.reset()

# Run the simulation with the trained agent
for _ in range(1500):
    # action, _ = model.predict(obs)
    action = env_agent.action_space.sample()  # Generate a random action
    obs, reward, done, _, _ = env_agent.step(action)  # Take a step in the environment

    env_display.step(action)  # Step in the display environment to show animation

    if done:
        obs, _ = env_agent.reset()  # Reset the agent's environment
        env_display.reset()  # Reset the display environment

# Close the environments
env_agent.close()
env_display.close()
