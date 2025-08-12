# main_training.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import MRzeroCore as mr0
import phantom_creator
import time
import pypulseq as pp
from simulate_and_process import simulate_and_process_mri
from plotting_utils import *
import os


# ===== SETUP PARAMETERS =====
# seq_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\23.7.25\epi_gre_mrf_epi_108\epi_gre_mrf_epi_no_inversion.seq"
phantom_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\most_updated\numerical_brain_cropped.mat"
# output_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\23.7.25\epi_gre_mrf_epi_108\run_14"

seq_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\6.8.25\epi_with_full_relaxation_in_calibration_phase\epi_gre_mrf_epi.seq"
output_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\6.8.25\epi_with_full_relaxation_in_calibration_phase"

# ===== CREATE OUTPUT FOLDERS =====
plots_output_path = os.path.join(output_path, 'plots3')
models_output_path = os.path.join(output_path, 'models')
os.makedirs(plots_output_path, exist_ok=True)
os.makedirs(models_output_path, exist_ok=True)

# ===== READ SEQUENCE =====
seq_pulseq = pp.Sequence()
seq_pulseq.read(seq_path)
Nx = int(seq_pulseq.get_definition('Nx'))
Ny = int(seq_pulseq.get_definition('Ny'))
flip_angles = seq_pulseq.get_definition('FlipAngles')
time_steps_number = len(flip_angles)
num_coils = 34

# ===== PLOTTING FLAG =====
plot = True

# ===== PREPARE PHANTOM =====
phantom, coil_maps = phantom_creator.create_phantom(Nx, Ny, phantom_path, num_coils=num_coils)
coil_maps = coil_maps.to("cuda")
obj_p = phantom.build()
reg_factors = np.logspace(np.log10(0.00001), np.log10(5), num=50)

diffs = []
for grappa_regularization_factor in reg_factors:
    calibration_data, time_series_shots, grappa_weights_torch = simulate_and_process_mri(obj_p, seq_path, num_coils, grappa_regularization_factor=grappa_regularization_factor)

    # Min/max for consistent grayscale scaling
    img_min = min(img.min() for img in time_series_shots)
    img_max = max(img.max() for img in time_series_shots)

    # Convert to numpy
    calib_img = calibration_data.squeeze().numpy()
    time_img = time_series_shots[0].detach().cpu().squeeze().numpy()
    diff_img = calib_img - time_img

    diffs.append(np.linalg.norm(diff_img))

    # Avoid divide-by-zero: add small epsilon where calib_img == 0
    epsilon = 1e-8
    norm_diff_img = diff_img / (calib_img + epsilon)

    # Create subplot with 4 images
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Calibration data
    # im1 = axes[0].imshow(calib_img, cmap='gray', vmin=img_min, vmax=img_max)
    im1 = axes[0].imshow(calib_img, cmap='gray', vmin=0, vmax=200)

    axes[0].set_title('Calibration Data')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Time series first shot
    # im2 = axes[1].imshow(time_img, cmap='gray', vmin=img_min, vmax=img_max)
    im2 = axes[1].imshow(time_img, cmap='gray', vmin=0, vmax=200)
    axes[1].set_title('Time Series Shot [0]')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Compute symmetric limits
    diff_max = np.abs(diff_img).max()
    norm_diff_max = np.abs(norm_diff_img).max()

    # Absolute difference
    im3 = axes[2].imshow(diff_img, cmap='seismic', vmin=-diff_max, vmax=diff_max)
    axes[2].set_title('Difference (Calib - Time[0])')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # Normalized difference
    im4 = axes[3].imshow(norm_diff_img, cmap='seismic', vmin=-norm_diff_max, vmax=norm_diff_max)
    axes[3].set_title('Normalized Diff ((Calib - Time[0]) / Calib)')
    axes[3].axis('off')
    plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(plots_output_path, f'calibration_reg__difference_{grappa_regularization_factor,diffs[-1]}.png'), dpi=150, bbox_inches='tight')
    # plt.savefig(os.path.join(plots_output_path, f'calibration_reg_{grappa_regularization_factor:.5f}_difference.png'), dpi=150, bbox_inches='tight')


plt.figure(figsize=(8, 6))
plt.scatter(reg_factors, diffs, color='blue', s=50)

plt.xscale('log')  # optional: use logarithmic scale for x-axis if your reg_factors span multiple orders
plt.xlabel('Regularization Factor')
plt.ylabel('Difference Metric')
plt.title('Effect of Regularization on Difference')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save or show the plot
plt.savefig(os.path.join(plots_output_path, 'reg_vs_diff.png'), dpi=150)



