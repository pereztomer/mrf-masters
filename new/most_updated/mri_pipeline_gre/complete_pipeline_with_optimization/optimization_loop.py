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
seq_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\23.7.25\epi_gre_mrf_epi_36\epi_gre_mrf_epi.seq"
phantom_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\most_updated\numerical_brain_cropped.mat"
output_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\23.7.25\epi_gre_mrf_epi_36\run_5"

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


model = create_vit_qmri_model(
    time_steps=len(time_series_shots),
    n_outputs=3,
    img_size=Nx,
    model_size="base",
    pretrained=True,  # Disable pretrained for comparison
    device='cuda'
)
# ===== OPTIMIZATION SETUP =====
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.99)
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

    t1_predicted = parameters_heat_map_predictions.squeeze()[0]  # .detach().requires_grad_()
    t2_predicted = parameters_heat_map_predictions.squeeze()[1]  # .detach().requires_grad_()
    pd_predicted = parameters_heat_map_predictions.squeeze()[2]  # .detach().requires_grad_()

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
    sim_calibration_img_sos, sim_images, _ = simulate_and_process_mri(obj_p_pred, seq_path, num_coils, grappa_weights_torch=grappa_weights_torch)
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

    if iteration % 3 == 0:

        fig = plt.figure(figsize=(24, 20))

        # Prepare data & vmin/vmax
        maps = {
            'T1': (T1_ground_truth, t1_predicted),
            'T2': (T2_ground_truth, t2_predicted),
            'PD': (PD_ground_truth, pd_predicted)
        }
        real_imgs = real_images_batch.squeeze().detach().cpu().numpy()
        sim_imgs = sim_images_batch.squeeze().detach().cpu().numpy()
        img_vmin, img_vmax = real_imgs.min(), real_imgs.max()
        img_vmin = min(img_vmin, sim_imgs.min())
        img_vmax = max(img_vmax, sim_imgs.max())

        # === Rows 1 & 2: GT and predicted maps ===
        for idx, (name, (gt, pred)) in enumerate(maps.items()):
            gt_np, pred_np = gt.squeeze().cpu().numpy(), pred.squeeze().detach().cpu().numpy()
            vmin, vmax = min(gt_np.min(), pred_np.min()), max(gt_np.max(), pred_np.max())

            for row, data, title in [(1, gt_np, f'Ground Truth {name}'),
                                     (2, pred_np, f'Predicted {name} (Iter {iteration})')]:
                ax = plt.subplot(4, 4, (row - 1) * 4 + idx + 1)
                im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
                ax.set_title(title)
                ax.axis('off')
                plt.colorbar(im, ax=ax)

        # Fill empty cells in cols 4
        for row in [1, 2]:
            ax = plt.subplot(4, 4, row * 4)
            ax.axis('off')

        # === Rows 3 & 4: time series images ===
        for t in range(min(time_steps_number, 4)):
            for row, imgs, label in [(3, real_imgs, 'Real'), (4, sim_imgs, 'Sim')]:
                ax = plt.subplot(4, 4, (row - 1) * 4 + t + 1)
                im = ax.imshow(imgs[t], cmap='gray', vmin=img_vmin, vmax=img_vmax)
                ax.set_title(f'{label} Image t={t}')
                ax.axis('off')
                plt.colorbar(im, ax=ax)

        plt.tight_layout()
        os.makedirs(os.path.join(plots_output_path, "iterations"), exist_ok=True)
        plot_path = os.path.join(plots_output_path, "iterations", f'iteration_{iteration:04d}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved iteration plot to {plot_path}")

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
