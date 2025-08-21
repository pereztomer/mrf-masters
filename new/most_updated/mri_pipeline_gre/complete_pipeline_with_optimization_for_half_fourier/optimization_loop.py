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
                          t1_pred, t2_pred, pd_pred, real_batch, sim_batch, plots_path,
                          t1_losses, t2_losses, pd_losses):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Current loss
    current_loss = losses[-1] if losses else 0

    # Main plot - back to 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle(f'Iteration {iteration + 1}/{epochs} | Loss: {current_loss:.6f}', fontsize=16)

    # Maps (rows 0-1)
    maps = [(T1_gt, t1_pred, 'T1'), (T2_gt, t2_pred, 'T2'), (PD_gt, pd_pred, 'PD')]

    for i, (gt, pred, name) in enumerate(maps):
        gt_np, pred_np = gt.cpu().numpy(), pred.detach().cpu().numpy()
        vmin, vmax = min(gt_np.min(), pred_np.min()), max(gt_np.max(), pred_np.max())

        # GT and Prediction
        for j, (img, title) in enumerate([(gt_np, f'GT {name}'), (pred_np, f'Pred {name}')]):
            im = axes[j, i].imshow(img, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[j, i].set_title(title)
            axes[j, i].axis('off')

        # Create colorbar on the right side
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axes[1, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    # Images (rows 2-3)
    real_imgs = real_batch.squeeze().detach().cpu().numpy()
    sim_imgs = sim_batch.squeeze().detach().cpu().numpy()
    vmin, vmax = min(real_imgs.min(), sim_imgs.min()), max(real_imgs.max(), sim_imgs.max())

    for t in range(4):
        for j, (imgs, prefix) in enumerate([(real_imgs, 'Real'), (sim_imgs, 'Sim')]):
            im = axes[j + 2, t].imshow(imgs[t * 5], cmap='gray', vmin=vmin, vmax=vmax)
            axes[j + 2, t].set_title(f'{prefix} t={t}')
            axes[j + 2, t].axis('off')

        # Create colorbar on the right side for each time point
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axes[3, t])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    # Hide unused subplots
    axes[0, 3].axis('off')
    axes[1, 3].axis('off')

    plt.tight_layout()
    os.makedirs(f"{plots_path}/iterations", exist_ok=True)
    plt.savefig(f"{plots_path}/iterations/iter_{iteration:04d}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Loss plots - separate subplot for each component
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Total loss
    axes[0, 0].semilogy(losses)
    axes[0, 0].set_title(f'Total Loss | Current: {current_loss:.6f}')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss (log scale)')
    axes[0, 0].grid(True, alpha=0.3)

    # Individual component losses
    components = [(t1_losses, 'T1', 'red'), (t2_losses, 'T2', 'blue'), (pd_losses, 'PD', 'green')]
    positions = [(0, 1), (1, 0), (1, 1)]

    for (losses_comp, name, color), (row, col) in zip(components, positions):
        current_comp_loss = losses_comp[-1] if losses_comp else 0
        axes[row, col].semilogy(losses_comp, color=color)
        axes[row, col].set_title(f'{name} Loss | Current: {current_comp_loss:.6f}')
        axes[row, col].set_xlabel('Iteration')
        axes[row, col].set_ylabel('Loss (log scale)')
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{plots_path}/loss_curve.png", dpi=150, bbox_inches='tight')
    plt.close()


# ===== SETUP PARAMETERS =====
seq_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\seq_11_8_25\gre_epi_108_fourier_factor_1.seq"
phantom_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\numerical_brain_cropped.mat"
output_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\seq_11_8_25\108\half_1\run_1"

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

obj_p = phantom.build()  # phantom for simulation

# ===== INITIAL SIMULATION DATA =====
calibration_data, time_series_shots, grappa_weights_torch = simulate_and_process_mri(obj_p, seq_path, num_coils)
grappa_weights_torch = grappa_weights_torch.detach()

T1_ground_truth = phantom.T1.squeeze().to("cuda")
T2_ground_truth = phantom.T2.squeeze().to("cuda")
PD_ground_truth = phantom.PD.squeeze().to("cuda")

# ===== CREATE MASK AND GROUND TRUTH =====
mask = T1_ground_truth > 0
mask = mask.to("cuda")

# Calculate normalization parameters
t1_mean = torch.mean(T1_ground_truth[mask])
t1_std = torch.std(T1_ground_truth[mask])
t2_mean = torch.mean(T2_ground_truth[mask])
t2_std = torch.std(T2_ground_truth[mask])
pd_mean = torch.mean(PD_ground_truth[mask])
pd_std = torch.std(PD_ground_truth[mask])

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
    model_size="tiny"  # Options: "tiny", "small", "medium", "large", "huge"
)

model = model.to("cuda")
# ===== OPTIMIZATION SETUP =====
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.99)

# ===== normelize images ========
n_valid = mask.sum()

# Apply mask and flatten spatial dimensions
masked_data = (time_series_shots * mask.unsqueeze(0)).view(50, -1)  # (50, 108*108)

# Calculate mean and std for each time step
means = masked_data.sum(dim=1) / n_valid  # (50,)
stds = torch.sqrt(((masked_data - means.unsqueeze(1))**2 * mask.view(1, -1)).sum(dim=1) / n_valid)  # (50,)

# Normalize each time step
time_series_shots_normalized = (time_series_shots - means.view(50, 1, 1)) / stds.view(50, 1, 1)

# ===== ADD BATCH DIMENSION =====
time_series_shots_normalized = time_series_shots_normalized.unsqueeze(0)  # shape: [1, n_TI, H, W]

# ===== MAIN TRAINING LOOP =====
print("Starting training...")
start_time = time.time()

losses = []
t1_losses = []
t2_losses = []
pd_losses = []
best_loos = 1000000000000
best_t1_loss = 1000000000000
best_t2_loss = 1000000000000
best_pd_loss = 1000000000000
for iteration in range(epochs):
    optimizer.zero_grad()

    # Forward
    parameters_heat_map_predictions = model(time_series_shots_normalized)

    t1_predicted = parameters_heat_map_predictions.squeeze()[0]
    t2_predicted = parameters_heat_map_predictions.squeeze()[1]
    pd_predicted = parameters_heat_map_predictions.squeeze()[2]

    # de-normalizing for mri simulator
    t1_predicted = t1_predicted * t1_std + t1_mean
    t2_predicted = t2_predicted * t2_std + t2_mean
    pd_predicted = pd_predicted * pd_std + pd_mean

    t1_predicted = t1_predicted * mask.float()
    t2_predicted = t2_predicted * mask.float()
    pd_predicted = pd_predicted * mask.float()

    current_t1_loss = F.mse_loss(t1_predicted, T1_ground_truth)
    current_t2_loss = F.mse_loss(t2_predicted, T2_ground_truth)
    current_pd_loss = F.mse_loss(pd_predicted, PD_ground_truth)

    t1_losses.append(current_t1_loss.item())
    t2_losses.append(current_t2_loss.item())
    pd_losses.append(current_pd_loss.item())

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

    losses.append(image_loss.item())
    if iteration % 100 == 0:
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        print(f"Grad norm: {total_grad_norm:.6f}")

    # Progress
    print(f"Iteration {iteration}: Loss = {image_loss.item():.8f}")

    # Plot results
    if (iteration % 50 == 0 or
            current_t1_loss < best_t1_loss or
            best_t2_loss < best_loos or
            best_pd_loss < best_pd_loss or
            image_loss < best_loos):
        # iteration, epochs, losses, T1_gt, T2_gt, PD_gt,
        # t1_pred, t2_pred, pd_pred, real_batch, sim_batch, plots_path,
        # t1_losses, t2_losses, pd_losses
        plot_training_results(iteration, epochs, losses, T1_ground_truth, T2_ground_truth, PD_ground_truth,
                              t1_predicted, t2_predicted, pd_predicted, time_series_shots, sim_images_batch,
                              plots_output_path, t1_losses, t2_losses, pd_losses)

    if current_t1_loss < best_t1_loss:
        best_t1_loss = current_t1_loss
    if current_t2_loss < best_t2_loss:
        best_t2_loss = current_t2_loss
    if current_pd_loss < best_pd_loss:
        best_pd_loss = current_pd_loss
    if image_loss < best_loos:
        best_loos = image_loss

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
