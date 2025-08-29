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
    # Get the data from the apd, downsample the data by factor, and save the data to a new folder
    # The new folder is called APD/original_folder_name
    # The data is saved as a numpy array
    for file in file_path:
        # LOAD FILES
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

def apd_load_main(file_path):
    # Load data from APD folder into dictionaries
    monitor_dict = {}
    apd_dict = {}
    params_dict = {}
    
    # Get folder name from path
    folder_name = os.path.basename(file_path)
    apd_path = os.path.join("APD", folder_name)
    
    # Load all transmission files
    pattern = os.path.join(apd_path, "*_transmission.npy")
    files = glob.glob(pattern)
    
    for file in files:
        base_name = os.path.basename(file).replace("_transmission.npy", "")
        apd_dict[base_name] = np.load(file)
        monitor_dict[base_name] = np.load(file.replace("_transmission.npy", "_monitor.npy"))
        
        # Load params
        params_file = file.replace("_transmission.npy", "_params.txt")
        with open(params_file, 'r') as f:
            params_dict[base_name] = f.read()
    
    return monitor_dict, apd_dict, params_dict



if __name__ == "__main__":
    # EXAMPLE USAGE
    path = "./Data/APD/2025.08.21 - Sample 13 Power Threshold"
    
    # Check if APD folder exists for this dataset
    folder_name = os.path.basename(path)
    apd_folder = os.path.join("APD", folder_name)
    
    if os.path.exists(apd_folder):
        monitor_dict, apd_dict, params_dict = apd_load_main(path)
        print(f"Loaded {len(apd_dict)} datasets successfully!")
    else:
        apd_main(path, factor=100)
        monitor_dict, apd_dict, params_dict = apd_load_main(path)
        print(f"Loaded {len(apd_dict)} datasets successfully!")






    
