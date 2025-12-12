#!/usr/bin/env python3
"""
Main entry point for nanostructure analysis package.

This script demonstrates how to import and use the nanostructure-analysis package.
"""

# There are two ways to use the package:

# Method 1: Import the entire package
import nanostructure_analysis as nsa

# Method 2: Import specific modules
# from nanostructure_analysis import apd_functions, confocal_functions
# from nanostructure_analysis.apd_plotting_functions import plot_apd_data

def main():
    print("Nanostructure Analysis Package")
    print(f"Version: {nsa.__version__}")
    print("Package successfully imported!")
    
    # Example usage would go here
    # data = nsa.load_data("path/to/data")
    # results = nsa.analyze_data(data)
    # nsa.plot_results(results)
    
    print("\nAvailable functions:")
    print("- APD functions: load_apd_data, analyze_apd_data, etc.")
    print("- Confocal functions: load_confocal_data, analyze_confocal_data, etc.")
    print("- SEM functions: load_sem_data, analyze_sem_data, etc.")
    print("- Spectra functions: load_spectra_data, analyze_spectra_data, etc.")
    print("- Plotting functions for each data type")

if __name__ == "__main__":
    main()
