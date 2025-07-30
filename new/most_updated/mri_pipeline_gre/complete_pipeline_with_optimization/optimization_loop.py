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


def plot_training_results(iteration, epochs, losses, T1_gt, T2_gt, PD_gt,
                          t1_pred, t2_pred, pd_pred, real_batch, sim_batch, plots_path):
    """Short plotting function for training visualization."""

    # Main plot
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))

    # Maps (rows 1-2)
    maps = [(T1_gt, t1_pred, 'T1'), (T2_gt, t2_pred, 'T2'), (PD_gt, pd_pred, 'PD')]
    for i, (gt, pred, name) in enumerate(maps):
        gt_np, pred_np = gt.cpu().numpy(), pred.detach().cpu().numpy()
        vmin, vmax = min(gt_np.min(), pred_np.min()), max(gt_np.max(), pred_np.max())

        axes[0, i].imshow(gt_np, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'GT {name}')
        axes[0, i].axis('off')

        axes[1, i].imshow(pred_np, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'Pred {name}')
        axes[1, i].axis('off')

    # Images (rows 3-4)
    real_imgs = real_batch.squeeze().detach().cpu().numpy()
    sim_imgs = sim_batch.squeeze().detach().cpu().numpy()
    vmin, vmax = min(real_imgs.min(), sim_imgs.min()), max(real_imgs.max(), sim_imgs.max())

    for t in range(4):
        axes[2, t].imshow(real_imgs[t * 5], cmap='gray', vmin=vmin, vmax=vmax)
        axes[2, t].set_title(f'Real t={t}')
        axes[2, t].axis('off')

        axes[3, t].imshow(sim_imgs[t * 5], cmap='gray', vmin=vmin, vmax=vmax)
        axes[3, t].set_title(f'Sim t={t}')
        axes[3, t].axis('off')

    # Hide unused subplots
    axes[0, 3].axis('off')
    axes[1, 3].axis('off')

    plt.tight_layout()
    os.makedirs(f"{plots_path}/iterations", exist_ok=True)
    plt.savefig(f"{plots_path}/iterations/iter_{iteration:04d}.png", dpi=150)
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.semilogy(losses)
    plt.title(f'Loss (Iter {iteration + 1}/{epochs}): {losses[-1]:.2e}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plots_path}/loss_curve.png", dpi=150)
    plt.close()

# ===== SETUP PARAMETERS =====
seq_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\23.7.25\epi_gre_mrf_epi_36\epi_gre_mrf_epi.seq"
phantom_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\most_updated\numerical_brain_cropped.mat"
output_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\23.7.25\epi_gre_mrf_epi_36\run_8"

epochs = 150

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
num_coils = 34
# ===== PLOTTING FLAG =====
plot = True

# ===== PREPARE PHANTOM =====
phantom, coil_maps = phantom_creator.create_phantom(Nx, Ny, phantom_path, num_coils=num_coils)
coil_maps = coil_maps.to("cuda")
T1_ground_truth = phantom.T1  # ground truth
T2_ground_truth = phantom.T2  # ground truth
PD_ground_truth = phantom.PD  # ground truth

t1_mean = torch.mean(T1_ground_truth)
t1_std = torch.std(T1_ground_truth)
t2_mean = torch.mean(T2_ground_truth)
t2_std = torch.std(T2_ground_truth)
pd_mean = torch.mean(PD_ground_truth)
pd_std = torch.std(PD_ground_truth)

obj_p = phantom.build()  # phantom for simulation

# ===== INITIAL SIMULATION DATA =====
calibration_data, time_series_shots, grappa_weights_torch = simulate_and_process_mri(obj_p, seq_path, num_coils)
grappa_weights_torch = grappa_weights_torch.detach()

mask = calibration_data > 100
mask = mask.to("cuda")

if plot:
    plot_phantom(phantom, save_path=os.path.join(plots_output_path, 'phantom.png'))
    display_time_series_shots(time_series_shots, flip_angles,
                              save_path=os.path.join(plots_output_path, 'time_series_shots.png'))

# ===== DEFINE NETWORK =====
from unet_3d_pre_trained import create_vit_qmri_model
from conv_unet import create_3d_unet_mri

model = create_3d_unet_mri(
    time_steps=len(time_series_shots),  # Your time dimension
    input_shape=(Nx, Nx),  # Your spatial dimensions
    out_channels=3,  # T1, T2, PD outputs
    model_size="large"  # Options: "tiny", "small", "medium", "large", "huge"
)

model = model.to("cuda")
# ===== OPTIMIZATION SETUP =====
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.99)
losses = []

# ===== normelize images ========
reshaped = time_series_shots.view(50, -1)  # [50, 1296]

# Calculate mean and std along the spatial dimensions (dim=1)
mean = reshaped.mean(dim=1, keepdim=True)  # [50, 1]
std = reshaped.std(dim=1, keepdim=True)  # [50, 1]

# Normalize
normalized_reshaped = (reshaped - mean) / (std + 1e-8)  # Add small epsilon to avoid div by zero

# Reshape back to original dimensions
time_series_shots_normalized = normalized_reshaped.view(50, 36, 36)

# ===== ADD BATCH DIMENSION =====
time_series_shots_normalized = time_series_shots_normalized.unsqueeze(0)  # shape: [1, n_TI, H, W]

# ===== MAIN TRAINING LOOP =====
print("Starting training...")
start_time = time.time()

for iteration in range(epochs):
    optimizer.zero_grad()

    # Forward
    parameters_heat_map_predictions = model(time_series_shots_normalized)

    t1_predicted = parameters_heat_map_predictions.squeeze()[0]
    t2_predicted = parameters_heat_map_predictions.squeeze()[1]
    pd_predicted = parameters_heat_map_predictions.squeeze()[2]

    # denormalizing for mri simulator
    t1_predicted = t1_predicted * t1_std + t1_mean
    t2_predicted = t2_predicted * t2_std + t2_mean
    pd_predicted = pd_predicted * pd_std + pd_mean

    t1_predicted = t1_predicted * mask.float()
    t2_predicted = t2_predicted * mask.float()
    pd_predicted = pd_predicted * mask.float()

    # Create phantom with predicted T1,T2 and PD
    obj_p_pred = phantom_creator.create_phantom_with_custom_parameters(
        T1_map=t1_predicted,
        T2_map=t2_predicted,
        PD_map=pd_predicted,
        Nread=Nx,
        Nphase=Ny,
        phantom_path=phantom_path,
        coil_maps=coil_maps
    )

    obj_p_pred = obj_p_pred.build()
    # Simulate images with predicted T1
    sim_calibration_img_sos, sim_images, _ = simulate_and_process_mri(obj_p_pred, seq_path, num_coils,
                                                                      grappa_weights_torch=grappa_weights_torch)
    sim_images_batch = sim_images.unsqueeze(0)


    # Compute loss
    image_loss = F.mse_loss(time_series_shots, sim_images_batch)

    # Backward
    image_loss.backward()
    # torch.nn.utils.clip_grad_norm_(T1_net.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    # Track loss
    losses.append(image_loss.item())

    # Progress
    print(f"Iteration {iteration}: Loss = {image_loss.item():.8f}")

    if iteration % 3 == 0:
        plot_training_results(iteration, epochs, losses, T1_ground_truth, T2_ground_truth, PD_ground_truth,
                              t1_predicted, t2_predicted, pd_predicted, time_series_shots, sim_images_batch, plots_output_path)



    # Early stopping
    if image_loss.item() < 1e-7:
        print(f"Converged at iteration {iteration}")
        break

# ===== FINAL RESULTS =====
total_time = time.time() - start_time
print(f"Training complete! Total time: {total_time:.2f} seconds")
print(f"Final loss: {losses[-1]:.8f}")

# ===== PLOT & SAVE LOSS CURVE =====
plot_training_results(iteration, epochs, losses, T1_ground_truth, T2_ground_truth, PD_ground_truth,
                              t1_predicted, t2_predicted, pd_predicted, time_series_shots, sim_images_batch, plots_output_path)

# ===== SAVE MODEL =====
model_save_path = os.path.join(models_output_path, 'best_t1_t2_pd_mapping_model.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_loss': losses[-1],
}, model_save_path)
print(f"Model saved to '{model_save_path}'")
