from nanostructure_analysis import *
import matplotlib.pyplot as plt












if __name__ == "__main__":
    apd_path = "./Data/APD/2025.08.21 - Sample 13 Power Threshold"
    apd_data = apd_load_main(apd_path)
    apd_data_box1 = filter_apd(apd_data, "*box1*")
    plot_apd(apd_data_box1)
    plt.show()
    plot_monitor(apd_data_box1)
    plt.show()
