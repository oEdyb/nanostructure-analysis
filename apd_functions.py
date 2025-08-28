import glob
import os
import numpy as np
import ast
import matplotlib.pyplot as plt

def get_apd_trace(file_path):
    # Get all apd traces in the folder, assuming the apd traces are in the format *_apd.dat
    apd_pattern = "*_transmission.npy"
    path = os.path.join(file_path, apd_pattern)
    path_to_apd = glob.glob(path)

    # Print the files that match the pattern
    print(f"\nFiles matching {apd_pattern}:")
    for file in path_to_apd:
        print(f"{os.path.basename(file)}")
    return path_to_apd
 


def get_apd_data(file_path, factor=100):
    # Get the data from the apd, and store it in a dictionary, preserving the filename as the key
    for file in file_path:
        apd_data = np.load(file)
        monitor_data = np.load(file.replace("_transmission.npy", "_monitor.npy"))
        # Change the data to be the average of every factor points
        n_points = len(apd_data) // factor * factor
        apd_chunked = apd_data[:n_points].reshape(-1, factor)
        monitor_chunked = monitor_data[:n_points].reshape(-1, factor)
        # Average each chunk
        apd_data = apd_chunked.mean(axis=1)  
        monitor_data = monitor_chunked.mean(axis=1)  
        base_name = os.path.basename(file).replace("_transmission.npy", "")

        # Extract the original folder name from the first file path
        original_folder = os.path.basename(os.path.dirname(file_path[0]))
        # Create APD/original_folder_name structure
        save_folder = os.path.join("APD", original_folder)
        os.makedirs(save_folder, exist_ok=True)
        print(f"Saving to: {save_folder}")

        apd_save_path = os.path.join(save_folder, f"{base_name}_transmission.npy")
        monitor_save_path = os.path.join(save_folder, f"{base_name}_monitor.npy")
        np.save(apd_save_path, apd_data)
        np.save(monitor_save_path, monitor_data)

         # Copy params file to new folder
        original_params_file = file.replace("_transmission.npy", "_params.txt")
        new_params_path = os.path.join(save_folder, f"{base_name}_params.txt")
        
        # Read and copy params file
        with open(original_params_file, 'r') as f:
            params_content = f.read()
        with open(new_params_path, 'w') as f:
            f.write(params_content)

    return 



def apd_main(file_path, factor=100):
    path_to_apd = get_apd_trace(file_path)    
    get_apd_data(path_to_apd, factor=factor)
    return

if __name__ == "__main__":
    apd_main("./Data/APD/2025.08.21 - Sample 13 Power Threshold", factor=1000)
