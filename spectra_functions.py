import glob
import os
import numpy as np
import ast
import matplotlib.pyplot as plt

def get_spectra(file_path):
    # Get all spectra in the folder, assuming the spectra are in the format *_spectrum.dat
    spectrum_pattern = "*_spectrum.dat"
    path = os.path.join(file_path, spectrum_pattern)
    path_to_spectra = glob.glob(path)

    # Print the files that match the pattern
    print(f"\nFiles matching {spectrum_pattern}:")
    for file in path_to_spectra:
        print(f"{os.path.basename(file)}")
    return path_to_spectra

def get_spectra_data(file_path):
    # Get the data from the spectra, and store it in a dictionary, preserving the filename as the key
    data_dict = {}
    for file in file_path:
        data = np.loadtxt(file)
        data_dict[os.path.basename(file).replace("_spectrum.dat", "")] = data
    print(f"Number of spectra: {len(data_dict.keys())}")
    return data_dict

def get_spectra_params(file_path):
    # Get the parameters from the spectra, and store it in a dictionary, preserving the filename as the key
    params_dict = {}
    for file in file_path:
        file = file.replace("_spectrum.dat", "_params.txt")
        # Read the dictionary from the text file and safely evaluate it, since it is in a dictionary format in my case
        with open(file, 'r') as f:
            params_text = f.read().strip()
            params = ast.literal_eval(params_text) 
        # Dictionary of dictionaries with the filename as the key
        params_dict[os.path.basename(file).replace("_params.txt", "")] = params
    print(f"Number of spectra: {len(params_dict.keys())}")
    return params_dict

def filter_spectra(data_dict, params_dict, pattern):
    # Use fnmatch to filter filenames directly
    filtered_data = {k: v for k, v in data_dict.items() if glob.fnmatch.fnmatch(k, pattern)}
    filtered_params = {k: v for k, v in params_dict.items() if glob.fnmatch.fnmatch(k, pattern)}
    
    print(f"Found {len(filtered_data)} files matching '{pattern}'")
    return filtered_data, filtered_params

def spectra_main(file_path, savgol_filter=True):
    path_to_spectra = get_spectra(file_path)
    data_dict = get_spectra_data(path_to_spectra)
    params_dict = get_spectra_params(path_to_spectra)
    return data_dict, params_dict


if __name__ == "__main__":

    # EXAMPLE USAGE
    data_dict, params_dict = spectra_main("./Data/Spectra/20250821 - sample13 - after")

    filtered_data, filtered_params = filter_spectra(data_dict, params_dict, "*box1*")

    for filename, data in filtered_data.items():
        plt.plot(data[:, 0], data[:, 1])
    plt.show()








