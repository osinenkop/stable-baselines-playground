import torch

# Specify the path to the saved output
file_path = "./cnn_outputs/cnn_output_step_40000.pt"
cnn_data = torch.load(file_path)
print(f"Loaded data: {type(cnn_data)}")

if isinstance(cnn_data, tuple):
    print(f"Tuple length: {len(cnn_data)}")
    for i, element in enumerate(cnn_data):
        print(f"Element {i}: Type={type(element)}, Shape={getattr(element, 'shape', 'N/A')}")
else:
    print(f"Unexpected format: {cnn_data}")
