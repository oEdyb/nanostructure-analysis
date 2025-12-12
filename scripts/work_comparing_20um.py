from nanostructure_analysis import *
import os

output_folder = "plots/big_aperatures"
os.makedirs(output_folder, exist_ok=True)

path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251009 - Sample 23"
path2 = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251110 - Sample 20"
spectra_data, spectra_params = spectra_main(path)
spectra_data_structure_5um, spectra_params_structure_5um = filter_spectra(spectra_data, spectra_params, "*um*", average=True, exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*D5*", "*A6*"])
spectra_data_2, spectra_params_2 = spectra_main(path2)
spectra_data_2, spectra_params_2 = filter_spectra(spectra_data_2, spectra_params_2, "*um*", average=True, exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*D5*", "*A6*"])
for key in spectra_data_structure_5um.keys():
    spectra_data_structure_5um[key][:, 1] = spectra_data_structure_5um[key][:, 1] / spectra_data_structure_5um[key][:, 1].max()

for key in spectra_data_2.keys():
    spectra_data_2[key][:, 1] = spectra_data_2[key][:, 1] / spectra_data_2[key][:, 1].max()
# spectra_data_structure_10um, spectra_params_structure_10um = filter_spectra(spectra_data, spectra_params, "10um*", average=True, exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*D5*", "*A6*"])
# spectra_data_structure_15um, spectra_params_structure_15um = filter_spectra(spectra_data, spectra_params, "15um*", average=True, exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*D5*", "*A6*"])
# spectra_data_structure_20um, spectra_params_structure_20um = filter_spectra(spectra_data, spectra_params, "20um*", average=True, exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*D5*", "*A6*"])

plot_spectra(spectra_data_structure_5um, spectra_params_structure_5um, cutoff=1000, new_fig=True)
plot_spectra(spectra_data_2, spectra_params_2, cutoff=1000, new_fig=False, linestyle='--')
# plot_spectra(spectra_data_structure_10um, spectra_params_structure_10um, cutoff=1000, new_fig=False)
# plot_spectra(spectra_data_structure_15um, spectra_params_structure_15um, cutoff=1000, new_fig=False)
# plot_spectra(spectra_data_structure_20um, spectra_params_structure_20um, cutoff=1000, new_fig=False)

plt.show()





