from apd_functions import * 
from spectra_functions import *
from plotting_functions import *
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

    Au_200nm_flag = False
    Au_100nm_flag = True

    output_folder = "plots/Au_200nm_vs_100nm"
    os.makedirs(output_folder, exist_ok=True)

    if Au_200nm_flag:
        # Create output folder for plots

        output_folder = "plots/Au_200nm_vs_100nm"
        os.makedirs(output_folder, exist_ok=True)
        
        # Load and process APD data
        apd_path = "./Data/APD/2025.08.21 - Sample 13 Power Threshold"
        apd_data, monitor_data, apd_params = apd_load_main(apd_path)
        apd_data_box1, monitor_data_box1, apd_params_box1 = filter_apd(apd_data, monitor_data, apd_params, "*box1*")
        
        # Plot and save APD data
        plot_apd(apd_data_box1, monitor_data_box1, apd_params_box1)
        plt.savefig(os.path.join(output_folder, "apd_data.png"))
        plt.close()

        # Plot and save APD data
        plot_apd(apd_data_box1, monitor_data_box1, apd_params_box1, time=150)
        plt.savefig(os.path.join(output_folder, "apd_data_150s.png"))
        plt.close()
        
        
        # Plot and save monitor data
        plot_monitor(apd_data_box1, monitor_data_box1, apd_params_box1)
        plt.savefig(os.path.join(output_folder, "monitor_data.png"))
        plt.close()

        # Load and process spectra data
        spectra_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250811 - sample6"
        spectra_path = "./Data/Spectra/20250821 - sample13 - after"
        spectra_data, spectra_params = load_spectra_cached(spectra_path)
        box1_data, box1_params = filter_spectra(spectra_data, spectra_params, "*box1*")
        spectra_data_5um, spectra_params_5um = filter_spectra(spectra_data, spectra_params, "*5um*_z_locked*")
        spectra_data_bkg, spectra_params_bkg = filter_spectra(spectra_data, spectra_params, "*bkg*")
        spectra_data_bkg_10000ms, spectra_params_bkg_10000ms = filter_spectra(spectra_data, spectra_params, "*bkg*_10000ms*")
        spectra_data_bkg_100ms, spectra_params_bkg_100ms = filter_spectra(spectra_data, spectra_params, "*bkg*_100ms*")

        # Normalize and plot spectra
        normalized_data = normalize_spectra(box1_data, spectra_data_bkg['bkg_10000ms_1'], spectra_data_5um['5um_100ms_z_locked_1'], spectra_data_bkg['bkg_100ms_3'])
        
        # Plot and save normalized spectra
        plot_spectra(box1_data, spectra_params)
        plt.gca().get_legend().remove()
        plt.savefig(os.path.join(output_folder, "spectra_box1.png"))
        plt.close()

        plot_spectra(normalized_data, spectra_params)
        plt.gca().get_legend().remove()
        plt.savefig(os.path.join(output_folder, "normalized_spectra.png"))
        plt.close()

        plot_spectra(spectra_data_bkg_10000ms, spectra_params_bkg_10000ms)
        plt.gca().get_legend().remove()
        plt.savefig(os.path.join(output_folder, "spectra_bkg_10000ms.png"))
        plt.close()
        
        # Plot and save 5um spectra
        plot_spectra(spectra_data_5um, spectra_params_5um)
        plt.gca().get_legend().remove()
        plt.savefig(os.path.join(output_folder, "spectra_5um.png"))
        plt.close()

        plot_spectra_transparent(box1_data, spectra_params)
        plt.savefig(os.path.join(output_folder, "spectra_box1_transparent.png"))
        plt.close()

        plot_spectra_transparent(normalized_data, spectra_params)
        plt.savefig(os.path.join(output_folder, "normalized_spectra_transparent.png"))
        plt.close()

        plot_spectra_transparent(spectra_data_5um, spectra_params_5um)
        plt.savefig(os.path.join(output_folder, "spectra_5um_transparent.png"))
        plt.close()
        
        plot_spectra_transparent(spectra_data_bkg_10000ms, spectra_params_bkg_10000ms)
        plt.savefig(os.path.join(output_folder, "spectra_bkg_10000ms_transparent.png"))
        plt.close()

        plot_spectra_transparent(spectra_data_bkg_100ms, spectra_params_bkg_100ms)
        plt.gca().set_ylim(0, 1200)  # Set y-axis limits before saving
        plt.savefig(os.path.join(output_folder, "spectra_bkg_100ms_transparent.png"))
        plt.close()



    if Au_100nm_flag:
        # Load and process spectra data
        spectra_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20250811 - sample6"
        spectra_data, spectra_params = load_spectra_cached(spectra_path)
        box1_data, box1_params = filter_spectra(spectra_data, spectra_params, "*box1*")
        spectra_data_5um, spectra_params_5um = filter_spectra(spectra_data, spectra_params, "*5um*_10ms*")
        spectra_data_bkg, spectra_params_bkg = filter_spectra(spectra_data, spectra_params, "*bkg*")
        spectra_data_bkg_5000ms, spectra_params_bkg_5000ms = filter_spectra(spectra_data, spectra_params, "*bkg*_5000ms*")
        spectra_data_bkg_100ms, spectra_params_bkg_100ms = filter_spectra(spectra_data, spectra_params, "*bkg*_10ms*")

        plot_spectra(box1_data, spectra_params)
        plt.savefig(os.path.join(output_folder, "100nm_spectra_box1.png"))
        plt.close()

        plot_spectra(spectra_data_5um, spectra_params_5um)
        plt.savefig(os.path.join(output_folder, "100nm_spectra_5um.png"))
        plt.close()

        plot_spectra(spectra_data_bkg, spectra_params_bkg)
        plt.savefig(os.path.join(output_folder, "100nm_spectra_bkg.png"))
        plt.close()

        plot_spectra(spectra_data_bkg_5000ms, spectra_params_bkg_5000ms)
        plt.savefig(os.path.join(output_folder, "100nm_spectra_bkg_5000ms.png"))
        plt.close()

        plot_spectra(spectra_data_bkg_100ms, spectra_params_bkg_100ms)
        plt.savefig(os.path.join(output_folder, "100nm_spectra_bkg_100ms.png"))
        plt.close()

        normalized_data = normalize_spectra(box1_data, spectra_data_bkg[bkg_key], spectra_data_5um[ref_key], spectra_data_bkg[ref_bkg_key])
