import MRzeroCore as mr0
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import random

# File paths
seq_file = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\progress\sequences\2025-05-15_epi_Nx_128_Ny_128_part_fourier_factor_1_R1_repetetions_1.seq"
phantom_file = "numerical_brain_cropped.mat"

plot_interpolations = True
# Load the sequence and phantom
seq0 = mr0.Sequence.import_file(seq_file)
obj_p = mr0.VoxelGridPhantom.load_mat(phantom_file)
obj_p = obj_p.build()

# Simulate the sequence
graph = mr0.compute_graph(seq0, obj_p, 200, 1e-3)
signal = mr0.execute_graph(graph, seq0, obj_p, print_progress=True)

# Reshape the signal
nFreqEnc = 184
time_points = signal.shape[0]
nAcq = time_points // nFreqEnc
rawdata = signal.numpy().reshape(nFreqEnc, nAcq)

# Extract k-space coordinates
ktraj_adc2 = seq0.get_kspace().numpy().reshape(nAcq, nFreqEnc, 4).transpose(2, 1, 0)
ky = ktraj_adc2[1, :, :].flatten()

# Calculate parameters
k_last, k_2last = ky[-1], ky[-2]
delta_ky = k_last - k_2last
fov = 1 / abs(delta_ky)

if k_last > 0:
    Ny_post = round(abs(k_last / delta_ky))
    Ny_pre = round(abs(np.min(ky) / delta_ky))
else:
    Ny_post = round(abs(np.max(ky) / delta_ky))
    Ny_pre = round(abs(k_last / delta_ky))

Nx = 2 * max(Ny_post, Ny_pre)
Ny = Nx
print(f"FOV: {fov:.3f} m, Nx: {Nx}, Ny: {Ny}")

kx_flat = ktraj_adc2[0].flatten()
kxmin, kxmax = min(kx_flat), max(kx_flat)
kxmax1 = kxmax / (Nx / 2 - 1) * (Nx / 2)
kmaxabs = max(kxmax1, -kxmin)
kxx = np.linspace(-Nx / 2, Nx / 2 - 1, Nx) / (Nx / 2) * kmaxabs

# Create interpolated data
interpolated_data = np.zeros((nAcq, len(kxx)), dtype=complex)

for acquisition_number in range(nAcq):
    specific_row = rawdata[:, acquisition_number]
    kspace_row_x = ktraj_adc2[0, :, acquisition_number]

    # Sort and handle duplicates
    sort_indices = np.argsort(kspace_row_x)
    kspace_row_x_sorted = kspace_row_x[sort_indices]
    specific_row_sorted = specific_row[sort_indices]

    unique_indices = np.unique(kspace_row_x_sorted, return_index=True)[1]
    kspace_row_x_unique = kspace_row_x_sorted[unique_indices]
    specific_row_unique = specific_row_sorted[unique_indices]

    # # Create spline and interpolate
    # cs = CubicSpline(kspace_row_x_unique, specific_row_unique, bc_type='linear', extrapolate=True)
    # interpolated_data[acquisition_number] = cs(kxx)

    # Linear interpolation
    interp_func = interp1d(kspace_row_x_unique, specific_row_unique,
                           kind='linear', bounds_error=False, fill_value=0)
    interpolated_data[acquisition_number] = interp_func(kxx)



print("here")
if plot_interpolations:
    # Plot for random rows
    num_rows = 3
    random_indices = random.sample(range(nAcq), num_rows)

    fig, axes = plt.subplots(num_rows, 2, figsize=(14, 4 * num_rows))

    for i, idx in enumerate(random_indices):
        row_data = rawdata[:, idx]
        kspace_x = ktraj_adc2[0, :, idx]
        interp_data = interpolated_data[idx]

        # Sort for plotting
        sort_idx = np.argsort(kspace_x)
        kspace_x_sorted = kspace_x[sort_idx]
        row_data_sorted = row_data[sort_idx]

        # Magnitude plot
        axes[i, 0].plot(kspace_x_sorted, np.abs(row_data_sorted), 'ro', label='Original')
        axes[i, 0].plot(kxx, np.abs(interp_data), 'b-', label='Interpolated')
        axes[i, 0].set_title(f'Acquisition {idx} - Magnitude')
        axes[i, 0].set_xlabel('k-space position (kx)')
        axes[i, 0].set_ylabel('Magnitude')
        axes[i, 0].grid(True)
        axes[i, 0].legend()

        # Phase plot
        axes[i, 1].plot(kspace_x_sorted, np.angle(row_data_sorted), 'ro', label='Original')
        axes[i, 1].plot(kxx, np.angle(interp_data), 'b-', label='Interpolated')
        axes[i, 1].set_title(f'Acquisition {idx} - Phase')
        axes[i, 1].set_xlabel('k-space position (kx)')
        axes[i, 1].set_ylabel('Phase (radians)')
        axes[i, 1].grid(True)
        axes[i, 1].legend()

    plt.tight_layout()
    plt.show()