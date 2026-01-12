import nanostructure_analysis as nsa
import matplotlib.pyplot as plt
import os
import re

output_dir = "plots/sample_20_WF_vs_Z"
os.makedirs(output_dir, exist_ok=True)

# Load WF data
wf_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251128 - Sample 20 WF No telescope"
wf_data, wf_params = nsa.spectra_main(wf_path)

# Load Z test data
z_test_path = r"\\AMIPC045962\Cache (D)\daily_data\spectra\20251104 - Sample 20 - Z tests"
z_data, z_params = nsa.spectra_main(z_test_path)

# Filter for z-position spectra
# WF dataset: 1um, -1um, 0um
wf_filtered, wf_params_filtered = nsa.filter_spectra(wf_data, wf_params, "*um*")
# Z test dataset: filter for 'z' in label
z_filtered, z_params_filtered = nsa.filter_spectra(z_data, z_params, "*z*")

print(f"WF spectra with 'z': {len(wf_filtered)}")
print(f"WF labels: {list(wf_filtered.keys())}")

print(f"\nZ test spectra with 'z': {len(z_filtered)}")
print(f"Z test labels: {list(z_filtered.keys())}")

# Plot WF spectra
plt.figure(figsize=(10, 6))
for label, spectrum in wf_filtered.items():
    if '0um' not in label.lower():
        plt.plot(spectrum[:, 0], spectrum[:, 1], label=label, alpha=0.7)
for label, spectrum in wf_filtered.items():
    if '0um' in label.lower():
        plt.plot(spectrum[:, 0], spectrum[:, 1], label=label, color='black', linewidth=2.5)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.title("WF")
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/wf.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot Confocal spectra - group by z0 and color-code
# Find all z0 labels and extract their base identifiers
z0_labels = [label for label in z_filtered.keys() if 'z0' in label.lower() or 'z_0' in label.lower()]

# Create groups based on base identifier (everything before z-position)
groups = {}
for z0_label in z0_labels:
    # Extract base by removing z-position part
    base = re.sub(r'[_-]?z[_-]?0.*', '', z0_label, flags=re.IGNORECASE)
    groups[base] = {'z0': z0_label, 'others': []}

# Assign other z-positions to their groups
for label in z_filtered.keys():
    if label not in z0_labels:
        for base in groups.keys():
            if label.startswith(base):
                groups[base]['others'].append(label)
                break

# Assign colors to each group
colors = plt.cm.tab10(range(len(groups)))

# Plot each group in separate figures
for idx, (base, group_data) in enumerate(groups.items()):
    plt.figure(figsize=(10, 6))
    color = colors[idx]

    # Plot other z-positions first (thin, transparent)
    for label in group_data['others']:
        plt.plot(z_filtered[label][:, 0], z_filtered[label][:, 1],
                color=color, alpha=0.3, linewidth=0.8, label=label)

    # Plot z0 on top (thick)
    z0_label = group_data['z0']
    plt.plot(z_filtered[z0_label][:, 0], z_filtered[z0_label][:, 1],
            label=z0_label, color=color, linewidth=2.5)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title(f"Confocal - {base}")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confocal_{base}.png", dpi=300, bbox_inches='tight')
    plt.show()
