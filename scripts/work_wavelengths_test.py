from nanostructure_analysis import *
import matplotlib.pyplot as plt
import os
import pickle


def load_spectra_cached(spectra_path):
    """Load spectra data with caching to avoid long reload times"""
    # Create cache directory
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache filename from path
    cache_name = spectra_path.replace("/", "_").replace("\\", "_").replace(":", "").replace(" ", "_")
    cache_file = os.path.join(cache_dir, f"spectra_{cache_name}.pkl")
    
    # Try loading from cache first
    if os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            print("Cache corrupted, reloading...")
    
    # Load fresh data if no cache
    print(f"Loading spectra from: {spectra_path}")
    spectra_data, spectra_params = spectra_main(spectra_path)
    
    # Save to cache
    print(f"Saving to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump((spectra_data, spectra_params), f)
    
    return spectra_data, spectra_params


def plot_max_vs_wavelength(image_dict):
    """Plot maximum intensity value vs wavelength extracted from filename"""
    import re
    
    # Extract wavelength and max values
    wavelengths = []
    max_values = []
    
    for key in image_dict.keys():
        # Extract wavelength from filename (looking for 3-digit number like 845, 852, etc.)
        match = re.search(r'(\d{3})', key)
        if match:
            wavelength = int(match.group(1))
            max_value = np.max(image_dict[key])
            wavelengths.append(wavelength)
            max_values.append(max_value)
    
    # Sort by wavelength
    sorted_data = sorted(zip(wavelengths, max_values))
    wavelengths, max_values = zip(*sorted_data)
    
    # Create plot with consistent styling
    plt.figure(figsize=(16, 8))
    plt.plot(wavelengths, max_values, 'o-', linewidth=3, markersize=10)
    
    plt.xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    plt.ylabel('Maximum Intensity', fontsize=18, fontweight='bold')
    plt.title('Maximum Intensity vs Wavelength', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()


if __name__ == "__main__":
    output_folder = "plots/wavelengths_test"
    os.makedirs(output_folder, exist_ok=True)

     # EXAMPLE USAGE WITH CACHE
    data_path = r"Data\Confocal\2025.06.11 - Wavelengths"
    image_dict, apd_dict, monitor_dict, xy_dict, z_dict = load_with_cache(data_path, confocal_main)
    print(image_dict.keys())

    plot_confocal(image_dict)
    plt.savefig(os.path.join(output_folder, "confocal_image.png"))


    plot_confocal(image_dict)
    plt.savefig(os.path.join(output_folder, "confocal.png"))

    
    # Plot max intensity vs wavelength
    plot_max_vs_wavelength(image_dict)
    plt.savefig(os.path.join(output_folder, "max_intensity_vs_wavelength.png"))
    plt.show()
    
