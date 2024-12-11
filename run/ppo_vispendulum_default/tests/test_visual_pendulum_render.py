import gymnasium as gym

# Initialize the standard Pendulum environment with "rgb_array" mode
env = gym.make("Pendulum-v1", render_mode="rgb_array")

# Reset the environment and attempt to render
obs, _ = env.reset()
image = env.render()

# Check if rendering produces a valid image
if image is None or image.size == 0:
    print("Render failed: Image is empty or None.")
else:
    print("Render succeeded: Image shape is", image.shape)

env.close()
