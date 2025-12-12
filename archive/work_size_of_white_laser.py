import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load image data
path = r"\\AMIPC045962\Cache (D)\daily_data\confocal_data\20250926 - Sample 20 Z tests\270nm_size_of_trapping_laser_y_axis_perp_pol_confocal_apd_traces_0015.npy"
data = np.load(path)

print(f"Original data shape: {data.shape}")

# Check if this is a time series or image stack
if len(data.shape) == 3:
    print(f"This is a 3D array: {data.shape[0]}x{data.shape[1]} spatial, {data.shape[2]} time points")
    # Average over time or take a specific frame
    image = np.mean(data, axis=2)  # Average over time dimension
    print(f"Using time-averaged image with shape: {image.shape}")
elif len(data.shape) == 2:
    image = data
    print(f"This is a 2D image with shape: {image.shape}")

print(f"Final image dimensions: height={image.shape[0]}, width={image.shape[1]}")
print("Note: numpy arrays are indexed as [row, column] = [y, x]")
print("So shape[0] = height (Y direction), shape[1] = width (X direction)")

# Define 2D Gaussian function
def gaussian_2d(xy, x0, y0, sigma_x, sigma_y, amplitude, offset):
    x, y = xy
    return amplitude * np.exp(-((x - x0)**2/(2*sigma_x**2) + (y - y0)**2/(2*sigma_y**2))) + offset

# Create coordinate grids
# Important: numpy uses [row, column] = [y, x] indexing
# mgrid creates y coordinates first (rows), then x coordinates (columns)
y, x = np.mgrid[:image.shape[0], :image.shape[1]]

print(f"Coordinate grid shapes: y={y.shape}, x={x.shape}")
print(f"Y goes from 0 to {image.shape[0]-1} (rows/height)")
print(f"X goes from 0 to {image.shape[1]-1} (columns/width)")

# Flatten arrays for fitting
x_flat = x.ravel()
y_flat = y.ravel()
z_flat = image.ravel()

# Initial parameter guesses
x0_guess = image.shape[1] / 2
y0_guess = image.shape[0] / 2
sigma_guess = min(image.shape) / 4
amplitude_guess = np.max(image)
offset_guess = np.min(image)

# Fit the Gaussian
popt, pcov = curve_fit(gaussian_2d, (x_flat, y_flat), z_flat,
                      p0=[x0_guess, y0_guess, sigma_guess, sigma_guess, amplitude_guess, offset_guess])

# Report fitted parameters
print(f"\nFitted Parameters:")
print(f"Center (x0, y0): ({popt[0]:.2f}, {popt[1]:.2f})")
print(f"  - x0 = {popt[0]:.2f} pixels (column position)")
print(f"  - y0 = {popt[1]:.2f} pixels (row position)")
print(f"Sigma (sigma_x, sigma_y): ({popt[2]:.2f}, {popt[3]:.2f})")
print(f"  - sigma_x = {popt[2]:.2f} pixels (width)")
print(f"  - sigma_y = {popt[3]:.2f} pixels (height)")
print(f"Amplitude: {popt[4]:.6f}")
print(f"Offset: {popt[5]:.6f}")

# Show which dimension is which
print(f"\nCoordinate System Check:")
print(f"Image shape: {image.shape} = (height={image.shape[0]}, width={image.shape[1]})")
print(f"Center is at column {popt[0]:.1f} (X) and row {popt[1]:.1f} (Y)")
if image.shape[0] != image.shape[1]:
    print(f"Image is rectangular: {'taller than wide' if image.shape[0] > image.shape[1] else 'wider than tall'}")

# Calculate physical beam size (1 pixel = 50 nm)
pixel_size_nm = 50
sigma_x_nm = popt[2] * pixel_size_nm
sigma_y_nm = popt[3] * pixel_size_nm

# Calculate FWHM (Full Width at Half Maximum)
fwhm_factor = 2 * np.sqrt(2 * np.log(2))  # ≈ 2.355
fwhm_x_nm = fwhm_factor * sigma_x_nm
fwhm_y_nm = fwhm_factor * sigma_y_nm

print("\nPhysical Beam Size (1 pixel = 50 nm):")
print(f"Beam width (σ_x): {sigma_x_nm:.1f} nm")
print(f"Beam height (σ_y): {sigma_y_nm:.1f} nm")
print(f"FWHM width: {fwhm_x_nm:.1f} nm")
print(f"FWHM height: {fwhm_y_nm:.1f} nm")
print(f"Beam area: {np.pi * sigma_x_nm * sigma_y_nm:.0f} nm²")

# Display the image with fitted Gaussian
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original image with fitted center
im1 = ax1.imshow(image, cmap='viridis')
ax1.plot(popt[0], popt[1], 'r+', markersize=12, label='Fitted center')
ax1.set_title('Original Image')
ax1.set_xlabel('X pixels')
ax1.set_ylabel('Y pixels')
ax1.legend()
plt.colorbar(im1, ax=ax1)

# Fitted Gaussian surface as contours
x_contour = np.linspace(0, image.shape[1]-1, image.shape[1])
y_contour = np.linspace(0, image.shape[0]-1, image.shape[0])
X_contour, Y_contour = np.meshgrid(x_contour, y_contour)

# Calculate fitted Gaussian values
fitted_gaussian = gaussian_2d((X_contour, Y_contour), *popt)

# Plot contours of fitted Gaussian
levels = np.linspace(np.min(fitted_gaussian), np.max(fitted_gaussian), 10)
contour = ax2.contour(X_contour, Y_contour, fitted_gaussian, levels=levels, cmap='plasma')
ax2.plot(popt[0], popt[1], 'r+', markersize=12, label='Fitted center')
ax2.set_title('Fitted 2D Gaussian Contours')
ax2.set_xlabel('X pixels')
ax2.set_ylabel('Y pixels')
ax2.legend()
plt.colorbar(contour, ax=ax2, label='Intensity')

plt.tight_layout()
plt.savefig('laser_beam_2d_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Create 1D profiles through the center
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Horizontal profile (X-direction) through center y
y_center = int(popt[1])  # Round to nearest pixel
x_line = np.arange(image.shape[1])
y_line = y_center
horizontal_profile = image[y_line, :]

# Plot horizontal profile
ax1.plot(x_line, horizontal_profile, 'b-', label='Data', linewidth=2)
# Calculate fitted Gaussian for this line
x_fit = np.linspace(0, image.shape[1]-1, 200)
y_fit = np.full_like(x_fit, popt[1])  # Fixed y position, same length as x_fit
horizontal_fit = popt[4] * np.exp(-((x_fit - popt[0])**2)/(2*popt[2]**2)) + popt[5]
ax1.plot(x_fit, horizontal_fit, 'r--', label='Gaussian fit', linewidth=2)
ax1.axvline(popt[0], color='g', linestyle=':', alpha=0.7, label='Center')
ax1.set_title(f'Horizontal Profile (Y = {y_center} pixel)')
ax1.set_xlabel('X position (pixels)')
ax1.set_ylabel('Intensity')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Vertical profile (Y-direction) through center x
x_center = int(popt[0])  # Round to nearest pixel
y_line = np.arange(image.shape[0])
vertical_profile = image[:, x_center]

# Plot vertical profile
ax2.plot(vertical_profile, y_line, 'b-', label='Data', linewidth=2)
# Calculate fitted Gaussian for this line
y_fit = np.linspace(0, image.shape[0]-1, 200)
x_fit = np.full_like(y_fit, popt[0])  # Fixed x position, same length as y_fit
vertical_fit = popt[4] * np.exp(-((y_fit - popt[1])**2)/(2*popt[3]**2)) + popt[5]
ax2.plot(vertical_fit, y_fit, 'r--', label='Gaussian fit', linewidth=2)
ax2.axhline(popt[1], color='g', linestyle=':', alpha=0.7, label='Center')
ax2.set_title(f'Vertical Profile (X = {x_center} pixel)')
ax2.set_xlabel('Intensity')
ax2.set_ylabel('Y position (pixels)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Physical units profiles (nm)
x_physical = np.arange(image.shape[1]) * pixel_size_nm
ax3.plot(x_physical, horizontal_profile, 'b-', label='Data', linewidth=2)
x_fit_physical = np.linspace(0, (image.shape[1]-1) * pixel_size_nm, 200)
ax3.plot(x_fit_physical, horizontal_fit, 'r--', label='Gaussian fit', linewidth=2)
ax3.axvline(popt[0] * pixel_size_nm, color='g', linestyle=':', alpha=0.7, label='Center')
ax3.set_title('Horizontal Profile (Physical Units)')
ax3.set_xlabel('X position (nm)')
ax3.set_ylabel('Intensity')
ax3.legend()
ax3.grid(True, alpha=0.3)

y_physical = np.arange(image.shape[0]) * pixel_size_nm
ax4.plot(vertical_profile, y_physical, 'b-', label='Data', linewidth=2)
y_fit_physical = np.linspace(0, (image.shape[0]-1) * pixel_size_nm, 200)
ax4.plot(vertical_fit, y_fit_physical, 'r--', label='Gaussian fit', linewidth=2)
ax4.axhline(popt[1] * pixel_size_nm, color='g', linestyle=':', alpha=0.7, label='Center')
ax4.set_title('Vertical Profile (Physical Units)')
ax4.set_xlabel('Intensity')
ax4.set_ylabel('Y position (nm)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('laser_beam_1d_profiles.png', dpi=150, bbox_inches='tight')
plt.show()

# Also show a 3D surface plot of the fitted Gaussian
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for 3D surface
x_3d = np.linspace(0, image.shape[1]-1, 30)
y_3d = np.linspace(0, image.shape[0]-1, 30)
X_3d, Y_3d = np.meshgrid(x_3d, y_3d)

# Calculate fitted Gaussian surface
Z_3d = gaussian_2d((X_3d, Y_3d), *popt)

# Plot 3D surface
surf = ax.plot_surface(X_3d, Y_3d, Z_3d, cmap='viridis', alpha=0.8)
ax.set_xlabel('X pixels')
ax.set_ylabel('Y pixels')
ax.set_zlabel('Intensity')
ax.set_title('3D Gaussian Fit Surface')
plt.colorbar(surf, ax=ax, shrink=0.8)
plt.savefig('laser_beam_3d_surface.png', dpi=150, bbox_inches='tight')
plt.show()