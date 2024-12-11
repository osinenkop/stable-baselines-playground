import os
import shutil

def clean_cnn_outputs(folder_path="./cnn_outputs"):
    """
    Deletes the contents of the specified folder. Creates the folder if it doesn't exist.

    Args:
        folder_path (str): The path of the folder to clean.
    """
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symlink
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(folder_path)
