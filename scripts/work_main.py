from nanostructure_analysis import *
import os
import matplotlib.pyplot as plt

import nanostructure_analysis.apd_functions as apd_functions
import nanostructure_analysis.spectra_functions as spectra_functions


if __name__ == "__main__":
    apd_path = "./Data/APD/2025.08.21 - Sample 13 Power Threshold"
    spectra_path = "./Data/Spectra/20250821 - sample13 - after"

    apd_data = apd_functions.apd_main(apd_path)
    spectra_data, spectra_params = spectra_functions.spectra_main(spectra_path)

    for label, trace in apd_data.transmission.items():
        plt.plot(trace, label=label)

    plt.legend()
    plt.show()


