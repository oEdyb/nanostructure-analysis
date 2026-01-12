import os
from nanostructure_analysis import *
import matplotlib.pyplot as plt
import re

output_folder = "plots/WF_vs_confocal_Z"
os.makedirs(output_folder, exist_ok=True)

legend_fontsize = 12

wf_path_1 = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251202 - Sample 20 Gold Far Off"
wf_path_2 = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251201 - Sample 20 - New Analysis"
wf_path_z = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251128 - Sample 20 WF No telescope"

confocal_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251113 - Sample 20 new reference holes"
confocal_path_z_1 = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251113 - Sample 20 new reference holes and white laser defocus"
confocal_path_z_2 = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251120 - Sample 20 White Laser Scan - High Signal"

# Load WF path 1 data
wf1_all, wf1_params_all = spectra_main(wf_path_1)
wf1_data, wf1_params = filter_spectra(wf1_all, wf1_params_all, "*[A-D][1-6]*", exclude=["*bias*", "*baseline*", "*um*", "*_0*", "*_1*", "*moving*"])
wf1_baseline, wf1_baseline_params = filter_spectra(wf1_all, wf1_params_all, "*gold*", average=False, exclude=["*bias*", "*baseline*", "*1um*", "*other*", "*pos7*"])
wf1_ref, wf1_ref_params = filter_spectra(wf1_all, wf1_params_all, "*30um*", average=False, exclude=["*bias*", "*baseline*", "*1um*"])

# Load WF path 2 data
wf2_all, wf2_params_all = spectra_main(wf_path_2)
wf2_data, wf2_params = filter_spectra(wf2_all, wf2_params_all, "*[A-D][1-6]*", exclude=["*bias*", "*baseline*", "*um*", "*_0*", "*_1*", "*moving*"])
wf2_baseline, wf2_baseline_params = filter_spectra(wf2_all, wf2_params_all, "*gold*pos*", average=False, exclude=["*bias*", "*baseline*", "*1um*", "*other*", "*pos7*"])
wf2_ref, wf2_ref_params = filter_spectra(wf2_all, wf2_params_all, "*30um*", average=False, exclude=["*bias*", "*baseline*", "*1um*"])

# Load WF path z data
wfz_all, wfz_params_all = spectra_main(wf_path_z)
wfz_data, wfz_params = filter_spectra(wfz_all, wfz_params_all, "*270nm*um*", exclude=["*bias*", "*baseline*", "*xy*"])
wfz_gold, wfz_gold_params = filter_spectra(wfz_all, wfz_params_all, "*gold*", average=False, exclude=["*bias*", "*baseline*", "*1um*"])

# Rename wfz labels
wfz_data_renamed = {}
wfz_params_renamed = {}
z_map = {"_0um": "1 um", "_1um": "0 um", "_-1um": "-1 um"}
for label, spectrum in wfz_data.items():
    new_label = label.replace("270nm", "").strip("_")
    for old, new in z_map.items():
        if old in label:
            new_label = new
            break
    wfz_data_renamed[new_label] = spectrum
    wfz_params_renamed[new_label] = wfz_params[label]
wfz_data, wfz_params = wfz_data_renamed, wfz_params_renamed

# Load confocal path data
conf_all, conf_params_all = spectra_main(confocal_path)
conf_data, conf_params = filter_spectra(conf_all, conf_params_all, "*[A-D][1-6]*", exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*bad*", "*270nm*", "*zlock*"])
conf_ref, conf_ref_params = filter_spectra(conf_all, conf_params_all, "*30um*", average=True, exclude=["*bias*", "*baseline*", "*270nm*"])

# Load confocal path z_1 data
conf_z1_all, conf_z1_params_all = spectra_main(confocal_path_z_1)
conf_z1_data, conf_z1_params = filter_spectra(conf_z1_all, conf_z1_params_all, "*[A-D]*white*", average=True, exclude=["*bias*", "*baseline*", "*100ms*", "*FVB*", "*bad*", "*270nm*", "*zlock*"])
conf_z1_ref, conf_z1_ref_params = filter_spectra(conf_z1_all, conf_z1_params_all, "*30um*_3*", average=False, exclude=["*bias*", "*baseline*", "*270nm*", "*30um_zlock_3*"])

# Load confocal path z_2 data
conf_z2_all, conf_z2_params_all = spectra_main(confocal_path_z_2)
conf_z2_3um, conf_z2_3um_params = filter_spectra(conf_z2_all, conf_z2_params_all, "*3um*", exclude=["*horizontal*", "*vertical*", "*baseline*", "*bias*"])
conf_z2_5um, conf_z2_5um_params = filter_spectra(conf_z2_all, conf_z2_params_all, "*5um*", exclude=["*horizontal*", "*vertical*", "*baseline*", "*bias*"])
conf_z2_30um, conf_z2_30um_params = filter_spectra(conf_z2_all, conf_z2_params_all, "*30um*", exclude=["*horizontal*", "*vertical*", "*baseline*", "*bias*"])
conf_z2_d1, conf_z2_d1_params = filter_spectra(conf_z2_all, conf_z2_params_all, "*A1*z*", exclude=["*horizontal*", "*vertical*", "*baseline*", "*bias*", "*last*"])

# Normalize WF path 1
wf1_normalized = normalize_spectra_baseline(wf1_data, wf1_baseline, wf1_ref, lam=1e7)

# Normalize WF path 2
wf2_normalized = normalize_spectra_baseline(wf2_data, wf2_baseline, wf2_ref, lam=1e7)

# Normalize WF z data
wfz_normalized = normalize_spectra(wfz_data, wfz_gold)

# Normalize confocal
conf_normalized = normalize_spectra(conf_data, conf_ref)

# Normalize confocal z2 d1 data
conf_z2_d1_normalized = normalize_spectra(conf_z2_d1, conf_z2_30um)

# Rename conf_z2_d1 labels to nm units (50 nm per z-step)
conf_z2_d1_renamed = {}
conf_z2_d1_params_renamed = {}
for label, spectrum in conf_z2_d1_normalized.items():
    # Extract z value from label (e.g., "A1_z0" -> 0, "A1_z5" -> 5, "A1_z-5" -> -5)
    match = re.search(r'z(-?\d+)', label)
    if match:
        z_value = int(match.group(1))
        nm_value = z_value * 50
        new_label = f"{nm_value} nm"
        conf_z2_d1_renamed[new_label] = spectrum
        conf_z2_d1_params_renamed[new_label] = conf_z2_d1_params[label]
    else:
        conf_z2_d1_renamed[label] = spectrum
        conf_z2_d1_params_renamed[label] = conf_z2_d1_params[label]
conf_z2_d1_normalized = conf_z2_d1_renamed
conf_z2_d1_params = conf_z2_d1_params_renamed

# Plot WF path 1
plot_spectra(wf1_normalized, wf1_params, cutoff=1000, new_fig=True, legend_fontsize=legend_fontsize)
plt.savefig(os.path.join(output_folder, "wf1_normalized.png"))
plt.close()

# Plot WF path 2
plot_spectra(wf2_normalized, wf2_params, cutoff=1000, new_fig=True, legend_fontsize=legend_fontsize)
plt.savefig(os.path.join(output_folder, "wf2_normalized.png"))
plt.close()

# Plot WF z data normalized
plot_spectra(wfz_normalized, wfz_params, cutoff=1000, new_fig=True, legend_fontsize=legend_fontsize)
plt.savefig(os.path.join(output_folder, "wfz_normalized.png"))
plt.close()

# Plot confocal data
plot_spectra(conf_data, conf_params, cutoff=1000, new_fig=True, legend_fontsize=legend_fontsize)
plt.savefig(os.path.join(output_folder, "conf.png"))
plt.close()

# Plot confocal z1 data
plot_spectra(conf_z1_data, conf_z1_params, cutoff=1000, new_fig=True, legend_fontsize=legend_fontsize)
plt.savefig(os.path.join(output_folder, "conf_z1.png"))
plt.close()

# Plot confocal z2 spot sizes
plot_spectra(conf_z2_3um, conf_z2_3um_params, cutoff=1000, new_fig=True, legend_fontsize=legend_fontsize)
plt.savefig(os.path.join(output_folder, "conf_z2_3um.png"))
plt.close()

plot_spectra(conf_z2_5um, conf_z2_5um_params, cutoff=1000, new_fig=True, legend_fontsize=legend_fontsize)
plt.savefig(os.path.join(output_folder, "conf_z2_5um.png"))
plt.close()

plot_spectra(conf_z2_30um, conf_z2_30um_params, cutoff=1000, new_fig=True, legend_fontsize=legend_fontsize)
plt.savefig(os.path.join(output_folder, "conf_z2_30um.png"))
plt.close()

plot_spectra(conf_z2_d1_normalized, conf_z2_d1_params, cutoff=1000, new_fig=True, legend_fontsize=legend_fontsize)
plt.savefig(os.path.join(output_folder, "conf_z2_d1_normalized.png"))
plt.close()

# Filter all D6 positions
wf1_d6, wf1_d6_params = filter_spectra(wf1_data, wf1_params, "*D6*")
wf2_d6, wf2_d6_params = filter_spectra(wf2_data, wf2_params, "*D6*")
conf_d6, conf_d6_params = filter_spectra(conf_data, conf_params, "*D6*")
conf_z1_d6, conf_z1_d6_params = filter_spectra(conf_z1_data, conf_z1_params, "*D6*")

wf1_d6_norm, wf1_d6_norm_params = filter_spectra(wf1_normalized, wf1_params, "*D6*")
wf2_d6_norm, wf2_d6_norm_params = filter_spectra(wf2_normalized, wf2_params, "*D6*")
conf_d6_norm, conf_d6_norm_params = filter_spectra(conf_normalized, conf_params, "*D6*")

# Add prefixes to avoid label collision
all_d6_raw = {f"WF1_{k}": v for k, v in wf1_d6.items()} | {f"WF2_{k}": v for k, v in wf2_d6.items()} | {f"Conf_{k}": v for k, v in conf_d6.items()} | {f"ConfZ1_{k}": v for k, v in conf_z1_d6.items()}
all_d6_raw_params = {f"WF1_{k}": v for k, v in wf1_d6_params.items()} | {f"WF2_{k}": v for k, v in wf2_d6_params.items()} | {f"Conf_{k}": v for k, v in conf_d6_params.items()} | {f"ConfZ1_{k}": v for k, v in conf_z1_d6_params.items()}
plot_spectra(all_d6_raw, all_d6_raw_params, cutoff=1000, new_fig=True, legend_fontsize=legend_fontsize)
plt.savefig(os.path.join(output_folder, "all_d6_raw.png"))
plt.close()

# Plot all D6 normalized
all_d6_norm = {f"WF1_{k}": v for k, v in wf1_d6_norm.items()} | {f"WF2_{k}": v for k, v in wf2_d6_norm.items()} | {f"Conf_{k}": v for k, v in conf_d6_norm.items()}
all_d6_norm_params = {f"WF1_{k}": v for k, v in wf1_d6_norm_params.items()} | {f"WF2_{k}": v for k, v in wf2_d6_norm_params.items()} | {f"Conf_{k}": v for k, v in conf_d6_norm_params.items()}
plot_spectra(all_d6_norm, all_d6_norm_params, cutoff=1000, new_fig=True, legend_fontsize=legend_fontsize)
plt.savefig(os.path.join(output_folder, "all_d6_normalized.png"))
plt.close()

# Plot WF1 vs Confocal D6 comparison
wf1_confocal_d6 = {}
wf1_confocal_d6_params = {}
for i, (k, v) in enumerate(wf1_d6_norm.items(), 1):
    label = "WF" if i == 1 else f"WF {i}"
    wf1_confocal_d6[label] = v
    wf1_confocal_d6_params[label] = wf1_d6_norm_params[k]
for i, (k, v) in enumerate(conf_d6_norm.items(), 1):
    label = "Confocal" if i == 1 else f"Confocal {i}"
    wf1_confocal_d6[label] = v
    wf1_confocal_d6_params[label] = conf_d6_norm_params[k]
plot_spectra(wf1_confocal_d6, wf1_confocal_d6_params, cutoff=1000, new_fig=True, legend_fontsize=legend_fontsize)
plt.savefig(os.path.join(output_folder, "wf1_vs_confocal_d6.png"))
plt.close()

# Plot A1 WF1 vs Confocal z0 comparison
wf1_a1, wf1_a1_params = filter_spectra(wf1_normalized, wf1_params, "*A1*")
conf_z0, conf_z0_params = filter_spectra(conf_z2_d1_normalized, conf_z2_d1_params, "*0 nm*")
a1_comparison = {"WF": list(wf1_a1.values())[0], "Confocal": list(conf_z0.values())[0]}
a1_comparison_params = {"WF": list(wf1_a1_params.values())[0], "Confocal": list(conf_z0_params.values())[0]}
plot_spectra(a1_comparison, a1_comparison_params, cutoff=1000, new_fig=True, legend_fontsize=legend_fontsize)
plt.savefig(os.path.join(output_folder, "wf1_a1_vs_confocal_z0.png"))
plt.close()


