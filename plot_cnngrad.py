import pandas as pd
import matplotlib.pyplot as plt

# Load the gradient log file
log_file = "gradients_log.csv"

# Read the CSV file into a DataFrame
# Columns in the CSV file: timestep, layer_name, grad_mean, grad_std
gradients_df = pd.read_csv(log_file, header=None, names=["timestep", "layer_name", "grad_mean", "grad_std"])

# List unique layers in the file
layers = gradients_df["layer_name"].unique()

# Create a plot for gradient mean and std for each layer
for layer in layers:
    layer_data = gradients_df[gradients_df["layer_name"] == layer]
    
    # Plot gradient mean
    plt.figure(figsize=(10, 5))
    plt.plot(layer_data["timestep"], layer_data["grad_mean"], label="Gradient Mean")
    plt.plot(layer_data["timestep"], layer_data["grad_std"], label="Gradient Std")
    plt.title(f"Gradients for Layer: {layer}")
    plt.xlabel("Timestep")
    plt.ylabel("Gradient Value")
    plt.legend()
    plt.grid(True)
    plt.show()
