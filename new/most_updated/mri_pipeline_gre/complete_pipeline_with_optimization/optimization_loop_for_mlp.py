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
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    """Short plotting function for training visualization."""
    # Main plot
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))

    # Maps (rows 1-2) with shared colorbars
    maps = [(T1_gt, t1_pred, 'T1'), (T2_gt, t2_pred, 'T2'), (PD_gt, pd_pred, 'PD')]
    for i, (gt, pred, name) in enumerate(maps):
        gt_np, pred_np = gt.cpu().numpy(), pred.detach().cpu().numpy()
        vmin, vmax = min(gt_np.min(), pred_np.min()), max(gt_np.max(), pred_np.max())

        # Ground truth
        im1 = axes[0, i].imshow(gt_np, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'GT {name}')
        axes[0, i].axis('off')

        # Prediction (same scale as GT)
        im2 = axes[1, i].imshow(pred_np, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'Pred {name}')
        axes[1, i].axis('off')

        # Single colorbar for both GT and Pred (shared scale)
        plt.colorbar(im1, ax=[axes[0, i], axes[1, i]], fraction=0.046, pad=0.04)

    # Images (rows 3-4) with shared colorbars
    real_imgs = real_batch.squeeze().detach().cpu().numpy()
    sim_imgs = sim_batch.squeeze().detach().cpu().numpy()
    vmin, vmax = min(real_imgs.min(), sim_imgs.min()), max(real_imgs.max(), sim_imgs.max())

    for t in range(4):
        # Real images
        im3 = axes[2, t].imshow(real_imgs[t * 5], cmap='gray', vmin=vmin, vmax=vmax)
        axes[2, t].set_title(f'Real t={t}')
        axes[2, t].axis('off')

        # Simulated images (same scale as real)
        im4 = axes[3, t].imshow(sim_imgs[t * 5], cmap='gray', vmin=vmin, vmax=vmax)
        axes[3, t].set_title(f'Sim t={t}')
        axes[3, t].axis('off')

        # Create colorbar for image pair
        divider = make_axes_locatable(axes[3, t])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im3, cax=cax)

    # Hide unused subplots
    axes[0, 3].axis('off')
    axes[1, 3].axis('off')

    plt.tight_layout()
    os.makedirs(f"{plots_path}/iterations", exist_ok=True)
    plt.savefig(f"{plots_path}/iterations/iter_{iteration:04d}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Enhanced Loss plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Full loss curve (log scale)
    axes[0].semilogy(losses)
    axes[0].set_title(f'Full Loss Curve (Iter {iteration + 1}/{epochs})')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss (log scale)')
    axes[0].grid(True, alpha=0.3)

    # Recent loss curve (linear scale for better detail)
    if len(losses) > 20:
        start_idx = max(0, len(losses) - min(50, len(losses) // 2))  # Last 50 iterations or half
        recent_losses = losses[start_idx:]
        recent_indices = range(start_idx, len(losses))

        axes[1].plot(recent_indices, recent_losses, 'b-', linewidth=2)
        axes[1].set_title(f'Recent Loss Detail (Last {len(recent_losses)} iters)')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Loss (linear scale)')
        axes[1].grid(True, alpha=0.3)

        # Add current loss value as text
        axes[1].text(0.05, 0.95, f'Current: {losses[-1]:.2e}',
                     transform=axes[1].transAxes, fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    else:
        axes[1].semilogy(losses)
        axes[1].set_title('Loss (Log Scale)')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Loss (log scale)')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{plots_path}/loss_curve.png", dpi=150, bbox_inches='tight')
    plt.close()


# ===== SETUP PARAMETERS =====
seq_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\23.7.25\epi_gre_mrf_epi_108\epi_gre_mrf_epi.seq"
phantom_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\most_updated\numerical_brain_cropped.mat"
output_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\23.7.25\epi_gre_mrf_epi_108\run_14"

# seq_path = "/home/tomer.perez/workspace/runs/gre_epi_108/epi_gre_mrf_epi_no_inversion.seq"s
# phantom_path = "/home/tomer.perez/workspace/data/numerical_brain_cropped.mat"
# output_path = "/home/tomer.perez/workspace/runs/gre_epi_108"
epochs = 1000

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

# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider
#
#
# # Interactive plot function
# def interactive_mask_plot():
#     gt_mask = T1_ground_truth > 0
#
#     fig, axes = plt.subplots(3, 3, figsize=(18, 12))
#     plt.subplots_adjust(bottom=0.15)
#
#     # Add slider
#     ax_slider = plt.axes([0.2, 0.02, 0.5, 0.03])
#     slider = Slider(ax_slider, 'Threshold', 1, 200, valinit=100, valfmt='%d')
#
#     def update(val):
#         threshold = slider.val
#         current_mask = calibration_data > threshold
#         missing = gt_mask.cpu() & ~current_mask.cpu()
#
#         # Clear and redraw
#         for ax in axes.flat:
#             ax.clear()
#
#         # Row 1: Masks
#         axes[0, 0].imshow(current_mask.cpu(), cmap='gray');
#         axes[0, 0].set_title('Current Mask')
#         axes[0, 1].imshow(gt_mask.cpu(), cmap='gray');
#         axes[0, 1].set_title('GT T1 Mask')
#         axes[0, 2].imshow(missing, cmap='gray');
#         axes[0, 2].set_title('Missing Pixels')
#
#         # Row 2: Data with masks
#         axes[1, 0].imshow((calibration_data.cpu() * current_mask.cpu()), cmap='gray');
#         axes[1, 0].set_title('Calib × Mask')
#         axes[1, 1].imshow((T1_ground_truth.cpu() * gt_mask.cpu()), cmap='viridis');
#         axes[1, 1].set_title('T1 × GT Mask')
#         axes[1, 2].axis('off')
#
#         # Row 3: Raw data
#         axes[2, 0].imshow(calibration_data.cpu(), cmap='gray');
#         axes[2, 0].set_title('Raw Calib')
#         axes[2, 1].imshow(T1_ground_truth.cpu(), cmap='viridis');
#         axes[2, 1].set_title('Raw T1')
#         axes[2, 2].axis('off')
#
#         plt.draw()
#
#     slider.on_changed(update)
#     update(100)
#     plt.show()
#
#
# interactive_mask_plot()

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
    im1 = axes[0].imshow(calib_img, cmap='gray', vmin=img_min, vmax=img_max)
    axes[0].set_title('Calibration Data')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Time series first shot
    im2 = axes[1].imshow(time_img, cmap='gray', vmin=img_min, vmax=img_max)
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

exit()
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

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005,           # Lower LR
    weight_decay=1e-4    # Add regularization for larger models
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


losses = []


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

    # # Create phantom and simulate
    obj_p_pred = phantom_creator.create_phantom_with_custom_parameters(
        T1_map=T1_ground_truth,
        T2_map=t2_predicted,
        PD_map=PD_ground_truth,
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

    # Calculate normalized loss using quantiles
    relative_loss = 0
    for t in range(len(time_series_shots)):
        # Extract masked pixels for this time point
        real_t = time_series_shots[t][mask]  # Shape: (num_masked_pixels,)
        sim_t = sim_images_batch.squeeze()[t][mask]  # Shape: (num_masked_pixels,)

        mse_loss = F.mse_loss(real_t, sim_t)
        mean = torch.mean(real_t)


        relative_loss += mse_loss/mean

    # Average across time points
    per_pixel_loss = relative_loss / len(time_series_shots)
    image_loss = per_pixel_loss


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

    # Progress
    print(f"Iteration {iteration}: Loss = {image_loss.item():.8f}")

    # Plot results
    if iteration % 3 == 0:
        plot_training_results(iteration, epochs, losses, T1_ground_truth, T2_ground_truth, PD_ground_truth,
                              t1_predicted, t2_predicted, pd_predicted, time_series_shots, sim_images_batch,
                              plots_output_path)

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