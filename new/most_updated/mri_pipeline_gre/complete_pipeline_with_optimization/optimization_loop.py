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
seq_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\20.7.25\epi_gre_mrf_epi_32\epi_gre_mrf_epi_32.seq"
phantom_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\most_updated\numerical_brain_cropped.mat"
output_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\20.7.25\epi_gre_mrf_epi_32"
epochs = 10000

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
num_coils = 4
# ===== PLOTTING FLAG =====
plot = True

# ===== PREPARE PHANTOM =====
phantom = phantom_creator.create_phantom(Nx, Ny, phantom_path, num_coils=num_coils)
T1_ground_truth = phantom.T1  # ground truth
T2_ground_truth = phantom.T2  # ground truth
PD_ground_truth = phantom.PD  # ground truth
obj_p = phantom.build()  # phantom for simulation

# ===== INITIAL SIMULATION DATA =====
calibration_data, time_series_shots = simulate_and_process_mri(obj_p, seq_path,num_coils)

if plot:
    plot_phantom(phantom, save_path=os.path.join(plots_output_path, 'phantom.png'))
    display_time_series_shots(time_series_shots, flip_angles,
                              save_path=os.path.join(plots_output_path, 'time_series_shots.png'))

# ===== DEFINE NETWORK =====
from unet_3d_pre_trained import create_vit_qmri_model

model = create_vit_qmri_model(time_steps=len(time_series_shots), n_outputs=3, img_size=Nx, pretrained=True, device='cuda')

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

    t1_predicted = parameters_heat_map_predictions.squeeze()[0] #.detach().requires_grad_()
    t2_predicted = parameters_heat_map_predictions.squeeze()[1] #.detach().requires_grad_()
    pd_predicted = parameters_heat_map_predictions.squeeze()[2] #.detach().requires_grad_()


    # Create phantom with predicted T1,T2 and PD
    obj_p_pred = phantom_creator.create_phantom_with_custom_parameters(
        T1_map=t1_predicted,
        T2_map=t2_predicted,
        PD_map=pd_predicted,
        Nread=Nx,
        Nphase=Ny,
        phantom_path=phantom_path,
        num_coils=num_coils
    )

    obj_p_pred = obj_p_pred.build()
    # Simulate images with predicted T1
    sim_calibration_img_sos, sim_images = simulate_and_process_mri(obj_p_pred, seq_path, num_coils)
    sim_images_batch = sim_images.unsqueeze(0)

    # Compute loss
    image_loss = F.mse_loss(real_images_batch, sim_images_batch)

    # Backward
    image_loss.backward()
    # torch.nn.utils.clip_grad_norm_(T1_net.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    # Track loss
    losses.append(image_loss.item())

    # Progress
    print(f"Iteration {iteration}: Loss = {image_loss.item():.8f}")

    if iteration % 50 == 0:
        fig = plt.figure(figsize=(24, 20))  # make taller for 4 rows

        # === Row 1: Ground truth maps ===
        ax1 = plt.subplot(4, 4, 1)
        im1 = ax1.imshow(T1_ground_truth.squeeze().cpu().numpy(), cmap='viridis')
        ax1.set_title('Ground Truth T1 Map')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1)

        ax2 = plt.subplot(4, 4, 2)
        im2 = ax2.imshow(T2_ground_truth.squeeze().cpu().numpy(), cmap='viridis')
        ax2.set_title('Ground Truth T2 Map')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)

        ax3 = plt.subplot(4, 4, 3)
        im3 = ax3.imshow(PD_ground_truth.squeeze().cpu().numpy(), cmap='viridis')
        ax3.set_title('Ground Truth PD Map')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3)

        # leave cell 4 empty or add something else
        ax4 = plt.subplot(4, 4, 4)
        ax4.axis('off')

        # === Row 2: Predicted maps ===
        ax5 = plt.subplot(4, 4, 5)
        im5 = ax5.imshow(t1_predicted.squeeze().detach().cpu().numpy(), cmap='viridis')
        ax5.set_title(f'Predicted T1 Map (Iter {iteration})')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5)

        ax6 = plt.subplot(4, 4, 6)
        im6 = ax6.imshow(t2_predicted.squeeze().detach().cpu().numpy(), cmap='viridis')
        ax6.set_title(f'Predicted T2 Map (Iter {iteration})')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6)

        ax7 = plt.subplot(4, 4, 7)
        im7 = ax7.imshow(pd_predicted.squeeze().detach().cpu().numpy(), cmap='viridis')
        ax7.set_title(f'Predicted PD Map (Iter {iteration})')
        ax7.axis('off')
        plt.colorbar(im7, ax=ax7)

        # leave cell 8 empty or add something else
        ax8 = plt.subplot(4, 4, 8)
        ax8.axis('off')

        # === Row 3: Real images time series ===
        real_imgs = real_images_batch.squeeze().detach().cpu().numpy()
        for t in range(min(time_steps_number, 4)):
            ax = plt.subplot(4, 4, 9 + t)  # slots 9,10,11,12
            im = ax.imshow(real_imgs[t], cmap='gray')
            ax.set_title(f'Real Image t={t}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)

        # === Row 4: Simulated / predicted images time series ===
        sim_imgs = sim_images_batch.squeeze().detach().cpu().numpy()
        for t in range(min(time_steps_number, 4)):
            ax = plt.subplot(4, 4, 13 + t)  # slots 13,14,15,16
            im = ax.imshow(sim_imgs[t], cmap='gray')
            ax.set_title(f'Sim Image t={t}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plot_filename = f'iteration_{iteration:04d}.png'
        plot_save_path = os.path.join(plots_output_path, plot_filename)
        plt.savefig(plot_save_path)
        plt.close()
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
model_save_path = os.path.join(models_output_path, 'best_t1_t2_pd_mapping_model.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_loss': losses[-1],
}, model_save_path)
print(f"Model saved to '{model_save_path}'")
