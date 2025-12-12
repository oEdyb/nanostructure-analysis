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


if __name__ == "__main__":
    output_folder = "plots/rods_polarization"
    os.makedirs(output_folder, exist_ok=True)

    # Load spectra data using the main path
    spectra_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250813 - sample20 - rods pol"
    spectra_data, spectra_params = load_spectra_cached(spectra_path)
    
    # Filter sample data
    sample_data, sample_params = filter_spectra(spectra_data, spectra_params, "[A-D]*horizontal*", average=True)
    sample_data_vertical, sample_params_vertical = filter_spectra(spectra_data, spectra_params, "[A-D]*vertical*", average=True)
    ref_data, ref_params = filter_spectra(spectra_data, spectra_params, "ref*", average=True)
    bkg_data, bkg_params = filter_spectra(spectra_data, spectra_params, "bkg*", average=True)
    um5_data, um5_params = filter_spectra(spectra_data, spectra_params, "5um*", average=True)
    
    # Normalize using specific file keys from the loaded data
    normalized_data = normalize_spectra_bkg(
        sample_data, 
        spectra_data['ref_horizontal_5000ms'], 
        spectra_data['5um_horizontal_50ms'], 
        spectra_data['ref_horizontal_50ms'], 
        savgol_before_bkg=True, 
        savgol_after_div=True, 
        savgol_after_div_window=131, 
        savgol_after_div_order=1
    )

    normalized_data_savgol = normalize_spectra_bkg(
        sample_data_vertical, 
        spectra_data['ref_vertical_5000ms'], 
        spectra_data['5um_vertical_50ms'], 
        spectra_data['ref_vertical_50ms'], 
        savgol_before_bkg=True, 
        savgol_after_div=True, 
        savgol_after_div_window=131, 
        savgol_after_div_order=1
    )
    
    # Plot and save normalized spectra
    plot_spectra(normalized_data, sample_params, cutoff=940)
    plt.savefig(os.path.join(output_folder, "normalized_spectra.png"))
    plt.close()

    plot_spectra(ref_data, ref_params, cutoff=940)
    plt.savefig(os.path.join(output_folder, "ref_spectra.png"))
    plt.close()

    plot_spectra(bkg_data, bkg_params, cutoff=940)
    plt.savefig(os.path.join(output_folder, "bkg_spectra.png"))
    plt.close()

    plot_spectra(um5_data, um5_params, cutoff=940)
    plt.savefig(os.path.join(output_folder, "um5_spectra.png"))
    plt.close()

    plot_spectra(normalized_data_savgol, sample_params_vertical, cutoff=940)
    plt.savefig(os.path.join(output_folder, "normalized_spectra_vertical.png"))
    plt.close()

    # Plot horizontal and vertical polarizations together for comparison
    plot_spectra(normalized_data, sample_params, cutoff=940, linestyle='-')
    plot_spectra(normalized_data_savgol, sample_params_vertical, cutoff=940, new_fig=False, linestyle='--')
    plt.savefig(os.path.join(output_folder, "normalized_spectra_horizontal_vs_vertical.png"))
    plt.close()
