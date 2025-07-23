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
seq_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\20.7.25\epi_gre_mrf_epi\epi_gre_mrf_epi.seq"
phantom_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\most_updated\numerical_brain_cropped.mat"
output_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\20.7.25\epi_gre_mrf_epi"
epochs = 100

# ===== CREATE OUTPUT FOLDERS =====
plots_output_path = os.path.join(output_path, 'plots')
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

# ===== PLOTTING FLAG =====
plot = True

# ===== PREPARE PHANTOM =====
phantom = phantom_creator.create_phantom(Nx, Ny, phantom_path, num_coils=34)
T1_ground_truth = phantom.T1  # ground truth
obj_p = phantom.build()  # phantom for simulation

# ===== INITIAL SIMULATION DATA =====
calibration_data, time_series_shots = simulate_and_process_mri(obj_p, seq_path)

if plot:
    plot_phantom(phantom, save_path=os.path.join(plots_output_path, 'phantom.png'))
    display_time_series_shots(time_series_shots, flip_angles,
                              save_path=os.path.join(plots_output_path, 'time_series_shots.png'))

# ===== DEFINE NETWORK =====
from unet_3d_pre_trained import create_vit_qmri_model

model = create_vit_qmri_model(time_steps=len(time_series_shots), n_outputs=3, pretrained=True)

# ===== OPTIMIZATION SETUP =====
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
losses = []

# ===== ADD BATCH DIMENSION =====
real_images_batch = time_series_shots.unsqueeze(0)  # shape: [1, n_TI, H, W]

# ===== MAIN TRAINING LOOP =====
print("Starting training...")
start_time = time.time()

for iteration in range(epochs):
    optimizer.zero_grad()

    # Forward
    parameters_heat_map_predictions = model(real_images_batch)

    # Create phantom with predicted T1
    obj_p_pred = phantom_creator.create_phantom_with_custom_parameters(
        T1_map=parameters_heat_map_predictions.squeeze()[0],
        T2_map=parameters_heat_map_predictions.squeeze()[1],
        PD_map=parameters_heat_map_predictions.squeeze()[2],
        Nread=Nx,
        Nphase=Ny,
        phantom_path=phantom_path
    )

    obj_p_pred = obj_p_pred.build()
    # Simulate images with predicted T1
    sim_calibration_img_sos, sim_images = simulate_and_process_mri(obj_p_pred, seq_path)
    sim_images_batch = sim_images.unsqueeze(0)

    # Compute loss
    image_loss = F.mse_loss(real_images_batch, sim_images_batch)

    # Backward
    image_loss.backward()
    torch.nn.utils.clip_grad_norm_(T1_net.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    # Track loss
    losses.append(image_loss.item())

    # Progress
    print(f"Iteration {iteration}: Loss = {image_loss.item():.8f}")

    # Plot and save every 15 iterations
    if iteration % 15 == 0:
        fig = plt.figure(figsize=(20, 12))

        # Ground truth T1 map
        ax1 = plt.subplot(3, 4, 1)
        im1 = ax1.imshow(T1_ground_truth.numpy(), cmap='viridis')
        ax1.set_title('Ground Truth T1 Map')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1)

        # Predicted T1 map
        ax2 = plt.subplot(3, 4, 2)
        T1_pred_display = T1_predicted.squeeze().detach().numpy()
        im2 = ax2.imshow(T1_pred_display, cmap='viridis')
        ax2.set_title(f'Predicted T1 Map (Iter {iteration})')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)

        # Real images time series
        real_imgs = real_images_batch.squeeze().detach().numpy()
        for t in range(time_steps_number):
            ax = plt.subplot(3, 4, 5 + t)
            im = ax.imshow(real_imgs[t], cmap='gray')
            ax.set_title(f'Real Image t={t}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)

        # Simulated images time series
        sim_imgs = sim_images_batch.squeeze().detach().numpy()
        for t in range(time_steps_number):
            ax = plt.subplot(3, 4, 9 + t)
            im = ax.imshow(sim_imgs[t], cmap='gray')
            ax.set_title(f'Sim Image t={t}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plot_filename = f'iteration_{iteration:04d}.png'
        plot_save_path = os.path.join(plots_output_path, plot_filename)
        plt.savefig(plot_save_path)
        plt.close()  # Free memory
        print(f"Saved iteration plot to {plot_save_path}")

    # Early stopping
    if image_loss.item() < 1e-7:
        print(f"Converged at iteration {iteration}")
        break

# ===== FINAL RESULTS =====
total_time = time.time() - start_time
print(f"Training complete! Total time: {total_time:.2f} seconds")
print(f"Final loss: {losses[-1]:.8f}")

# ===== PLOT & SAVE LOSS CURVE =====
plt.figure(figsize=(10, 6))
plt.semilogy(losses)
plt.title('Training Loss vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('MSE Loss (Log Scale)')
plt.grid(True)
plt.tight_layout()
loss_curve_path = os.path.join(plots_output_path, 'training_loss_curve.png')
plt.savefig(loss_curve_path)
plt.close()
print(f"Loss curve saved to {loss_curve_path}")

# ===== SAVE MODEL =====
model_save_path = os.path.join(models_output_path, 'best_t1_mapping_model.pth')
torch.save({
    'model_state_dict': T1_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_loss': losses[-1],
    'network_type': 'T1MappingNet_ResNet18'
}, model_save_path)
print(f"Model saved to '{model_save_path}'")
