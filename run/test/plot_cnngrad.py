import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file and assign proper headers
gradient_data = pd.read_csv(
    "gradients_log.csv",
    header=None,  # Since the file has no proper header, we assign it manually
    names=["Timestep", "Layer", "Gradient Mean", "Gradient Std"]
)

# Ensure no hidden characters in the Layer column
gradient_data["Layer"] = gradient_data["Layer"].str.strip()

# Detect episode resets by identifying when the timestep decreases
gradient_data["Episode"] = (gradient_data["Timestep"] < gradient_data["Timestep"].shift(1)).cumsum()

# Calculate a global timestep offset for each episode
episode_offsets = (
    gradient_data.groupby("Episode")["Timestep"].max().cumsum().shift(fill_value=0)
)

# Add the offsets to calculate the true global timestep
gradient_data["Global Timestep"] = (
    gradient_data["Timestep"] + gradient_data["Episode"].map(episode_offsets)
)

# Extract unique layers
unique_layers = gradient_data["Layer"].unique()

# Iterate through each layer to plot gradients
for layer in unique_layers:
    # Filter out only data for the current layer
    layer_data = gradient_data[gradient_data["Layer"] == layer]
    
    plt.figure()
    plt.plot(
        layer_data["Global Timestep"], 
        layer_data["Gradient Mean"], 
        label="Gradient Mean"
    )
    plt.plot(
        layer_data["Global Timestep"], 
        layer_data["Gradient Std"], 
        label="Gradient Std"
    )
    plt.xlabel("Global Timestep")
    plt.ylabel("Gradient Value")
    plt.title(f"Gradients for Layer: {layer}")
    plt.legend()
    plt.grid(True)  # Optional: Add a grid for better readability
    plt.show()
