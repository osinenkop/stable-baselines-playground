# Plot learning curve

import pandas as pd
import matplotlib.pyplot as plt

# Load the episode rewards from the CSV file
data = pd.read_csv("logs/episode_rewards.csv")

# Plot the learning curve
plt.figure(figsize=(10, 5))
plt.plot(data['Episode'], data['Reward'], label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()
