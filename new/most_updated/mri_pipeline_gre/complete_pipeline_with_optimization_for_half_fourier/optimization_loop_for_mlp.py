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


# def plot_training_results(iteration, epochs, losses, T1_gt, T2_gt, PD_gt,
#                           t1_pred, t2_pred, pd_pred, real_batch, sim_batch, plots_path):
#     from mpl_toolkits.axes_grid1 import make_axes_locatable
#     """Short plotting function for training visualization."""
#     # Main plot
#     fig, axes = plt.subplots(4, 4, figsize=(20, 16))
#
#     # Maps (rows 1-2) with shared colorbars
#     maps = [(T1_gt, t1_pred, 'T1'), (T2_gt, t2_pred, 'T2'), (PD_gt, pd_pred, 'PD')]
#     for i, (gt, pred, name) in enumerate(maps):
#         gt_np, pred_np = gt.cpu().numpy(), pred.detach().cpu().numpy()
#         vmin, vmax = min(gt_np.min(), pred_np.min()), max(gt_np.max(), pred_np.max())
#
#         # Ground truth
#         im1 = axes[0, i].imshow(gt_np, cmap='viridis', vmin=vmin, vmax=vmax)
#         axes[0, i].set_title(f'GT {name}')
#         axes[0, i].axis('off')
#
#         # Prediction (same scale as GT)
#         im2 = axes[1, i].imshow(pred_np, cmap='viridis', vmin=vmin, vmax=vmax)
#         axes[1, i].set_title(f'Pred {name}')
#         axes[1, i].axis('off')
#
#         # Single colorbar for both GT and Pred (shared scale)
#         plt.colorbar(im1, ax=[axes[0, i], axes[1, i]], fraction=0.046, pad=0.04)
#
#     # Images (rows 3-4) with shared colorbars
#     real_imgs = real_batch.squeeze().detach().cpu().numpy()
#     sim_imgs = sim_batch.squeeze().detach().cpu().numpy()
#     vmin, vmax = min(real_imgs.min(), sim_imgs.min()), max(real_imgs.max(), sim_imgs.max())
#
#     for t in range(4):
#         # Real images
#         im3 = axes[2, t].imshow(real_imgs[t * 5], cmap='gray', vmin=vmin, vmax=vmax)
#         axes[2, t].set_title(f'Real t={t}')
#         axes[2, t].axis('off')
#
#         # Simulated images (same scale as real)
#         im4 = axes[3, t].imshow(sim_imgs[t * 5], cmap='gray', vmin=vmin, vmax=vmax)
#         axes[3, t].set_title(f'Sim t={t}')
#         axes[3, t].axis('off')
#
#         # Create colorbar for image pair
#         divider = make_axes_locatable(axes[3, t])
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         plt.colorbar(im3, cax=cax)
#
#     # Hide unused subplots
#     axes[0, 3].axis('off')
#     axes[1, 3].axis('off')
#
#     plt.tight_layout()
#     os.makedirs(f"{plots_path}/iterations", exist_ok=True)
#     plt.savefig(f"{plots_path}/iterations/iter_{iteration:04d}.png", dpi=150, bbox_inches='tight')
#     plt.close()
#
#     # Enhanced Loss plots
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))
#
#     # Full loss curve (log scale)
#     axes[0].semilogy(losses)
#     axes[0].set_title(f'Full Loss Curve (Iter {iteration + 1}/{epochs})')
#     axes[0].set_xlabel('Iteration')
#     axes[0].set_ylabel('Loss (log scale)')
#     axes[0].grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig(f"{plots_path}/loss_curve.png", dpi=150, bbox_inches='tight')
#     plt.close()
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
# seq_path = r"/home/tomer.perez/workspace/runs/gre_epi_72/gre_epi_72.seq"
# phantom_path = r"/home/tomer.perez/workspace/data/numerical_brain_cropped.mat"
# output_path = r"/home/tomer.perez/workspace/runs/gre_epi_72/run_1"

# seq_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\seq_11_8_25\gre_epi_128_fourier_factor_0.875.seq"
# phantom_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\numerical_brain_cropped.mat"
# output_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\seq_11_8_25\128\half_0.875\run_1"

seq_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\seq_11_8_25\gre_epi_108_fourier_factor_1.seq"
phantom_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\numerical_brain_cropped.mat"
output_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\seq_11_8_25\108\half_1\run_1"

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
num_coils = 34

# ===== PLOTTING FLAG =====
plot = True

# ===== PREPARE PHANTOM =====
phantom, coil_maps = phantom_creator.create_phantom(Nx, Ny, phantom_path, num_coils=num_coils)
coil_maps = coil_maps.to("cuda")
obj_p = phantom.build()

# ===== INITIAL SIMULATION DATA =====
calibration_data, time_series_shots, grappa_weights_torch = simulate_and_process_mri(obj_p, seq_path, num_coils)
grappa_weights_torch = grappa_weights_torch.detach()

T1_ground_truth = phantom.T1.squeeze().to("cuda")
T2_ground_truth = phantom.T2.squeeze().to("cuda")
PD_ground_truth = phantom.PD.squeeze().to("cuda")

# ===== CREATE MASK AND GROUND TRUTH =====
# mask = calibration_data > 150
mask = T1_ground_truth > 0
mask = mask.to("cuda")

# Calculate normalization parameters
t1_mean = torch.mean(T1_ground_truth[mask])
t1_std = torch.std(T1_ground_truth[mask])
t2_mean = torch.mean(T2_ground_truth[mask])
t2_std = torch.std(T2_ground_truth[mask])
pd_mean = torch.mean(PD_ground_truth[mask])
pd_std = torch.std(PD_ground_truth[mask])

# ===== PREPARE DATA FOR TRAINING =====
# Get masked pixel time series
mask_expanded = mask.unsqueeze(0).expand_as(time_series_shots)
masked_time_series = time_series_shots[mask_expanded]
num_masked_pixels = torch.sum(mask).item()
masked_reshaped = masked_time_series.view(time_steps_number, num_masked_pixels)

# Normalize time series
means = torch.mean(masked_reshaped, dim=1, keepdim=True)
stds = torch.std(masked_reshaped, dim=1, keepdim=True)
normalized_time_series = (masked_reshaped - means) / (stds + 1e-8)

# Transpose to get (num_pixels, time_steps) format
pixel_time_series = normalized_time_series.transpose(0, 1).to("cuda")  # Shape: (665, 50)

# Get spatial indices for reconstruction
masked_indices = torch.where(mask)
masked_rows = masked_indices[0]
masked_cols = masked_indices[1]

if plot:
    plot_phantom(phantom, save_path=os.path.join(plots_output_path, 'phantom.png'))
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Min/max for consistent grayscale scaling
    img_min = min(img.min() for img in time_series_shots)
    img_max = max(img.max() for img in time_series_shots)

    # Convert to numpy
    calib_img = calibration_data.squeeze().numpy()
    time_img = time_series_shots[0].detach().cpu().squeeze().numpy()
    diff_img = calib_img - time_img

    # Avoid divide-by-zero: add small epsilon where calib_img == 0
    epsilon = 1e-8
    norm_diff_img = diff_img / (calib_img + epsilon)

    # Create subplot with 4 images
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Calibration data
    # im1 = axes[0].imshow(calib_img, cmap='gray', vmin=img_min, vmax=img_max)
    im1 = axes[0].imshow(calib_img, cmap='gray')
    axes[0].set_title('Calibration Data')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Time series first shot
    # im2 = axes[1].imshow(time_img, cmap='gray', vmin=img_min, vmax=img_max)
    im2 = axes[1].imshow(time_img, cmap='gray')
    axes[1].set_title('Time Series Shot [0]')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Absolute difference
    im3 = axes[2].imshow(diff_img, cmap='RdBu_r')
    axes[2].set_title('Difference (Calib - Time[0])')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # Normalized difference
    im4 = axes[3].imshow(norm_diff_img, cmap='RdBu_r')
    axes[3].set_title('Normalized Diff ((Calib - Time[0]) / Calib)')
    axes[3].axis('off')
    plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_path, 'calibration5.png'), dpi=150, bbox_inches='tight')

    display_time_series_shots(time_series_shots, flip_angles,
                              save_path=os.path.join(plots_output_path, 'time_series_shots.png'))

# ===== DEFINE NETWORK =====
from mlp import create_simple_mlp

model = create_simple_mlp(
    input_features=time_steps_number,  # 50 time steps
    output_features=3,  # T1, T2, PD
    model_size="huge+"
)
model = model.to("cuda")

# ===== OPTIMIZATION SETUP =====
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Higher LR
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # Less aggressive

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

losses = []
t1_losses = []
t2_losses = []
pd_losses = []

best_loos = 1000000000000
best_t1_loss = 1000000000000
best_t2_loss = 1000000000000
best_pd_loss = 1000000000000

# ===== MAIN TRAINING LOOP =====
print("Starting training...")
start_time = time.time()

for iteration in range(epochs):
    optimizer.zero_grad()

    # Forward pass: all pixels at once
    predictions = model(pixel_time_series)  # Shape: (665, 3)

    # Reconstruct spatial maps
    t1_predicted = torch.zeros_like(T1_ground_truth).to("cuda")
    t2_predicted = torch.zeros_like(T2_ground_truth).to("cuda")
    pd_predicted = torch.zeros_like(PD_ground_truth).to("cuda")

    # Fill in predictions at masked locations
    t1_predicted[masked_rows, masked_cols] = predictions[:, 0]
    t2_predicted[masked_rows, masked_cols] = predictions[:, 1]
    pd_predicted[masked_rows, masked_cols] = predictions[:, 2]

    # Denormalize
    t1_predicted = t1_predicted * t1_std + t1_mean
    t2_predicted = t2_predicted * t2_std + t2_mean
    pd_predicted = pd_predicted * pd_std + pd_mean

    # Apply mask
    t1_predicted = t1_predicted * mask.float()
    t2_predicted = t2_predicted * mask.float()
    pd_predicted = pd_predicted * mask.float()

    current_t1_loss = F.mse_loss(t1_predicted, T1_ground_truth)
    current_t2_loss = F.mse_loss(t2_predicted, T2_ground_truth)
    current_pd_loss = F.mse_loss(pd_predicted, PD_ground_truth)

    t1_losses.append(current_t1_loss.item())
    t2_losses.append(current_t2_loss.item())
    pd_losses.append(current_pd_loss.item())

    # # Create phantom and simulate
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

    # Simulate images
    sim_calibration_img_sos, sim_images, _ = simulate_and_process_mri(
        obj_p_pred, seq_path, num_coils, grappa_weights_torch=grappa_weights_torch
    )
    sim_images_batch = sim_images.unsqueeze(0)

    # Calculate masked loss (only for brain regions)
    mask_expanded = mask.unsqueeze(0).expand_as(time_series_shots)  # Shape: (50, 36, 36)

    image_loss = F.mse_loss(time_series_shots[mask_expanded], sim_images_batch.squeeze()[mask_expanded])


    def check_gradients(model, loss):
        print(f"Loss requires grad: {loss.requires_grad}")
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"{name}: grad_norm = {grad_norm:.8f}")
            else:
                print(f"{name}: NO GRADIENT!")


    # Backward pass
    image_loss.backward()
    # check_gradients(model, image_loss)

    optimizer.step()
    scheduler.step()
    # Track loss
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
