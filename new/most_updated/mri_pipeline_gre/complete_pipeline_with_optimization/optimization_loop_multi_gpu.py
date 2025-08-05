# main_training.py - Complete Multi-GPU Version

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

        # Create colorbar between the two axes
        divider = make_axes_locatable(axes[1, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)

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


class SimpleMultiGPU:
    def __init__(self, base_model, num_splits):
        self.base_model = base_model
        self.num_splits = num_splits

        # Check GPU availability
        available_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {available_gpus}")

        # Validate configuration
        if num_splits not in [2, 4]:
            print(f"ERROR: num_splits must be 2 or 4, got {num_splits}")
            exit(1)

        if available_gpus < num_splits:
            print(f"ERROR: Need {num_splits} GPUs but only {available_gpus} available")
            exit(1)

        self.devices = [f"cuda:{i}" for i in range(num_splits)]

        # Create model copies on different GPUs
        self.models = []
        for device in self.devices:
            model_copy = type(base_model)(
                input_features=base_model.input_features if hasattr(base_model, 'input_features') else 50,
                output_features=base_model.output_features if hasattr(base_model, 'output_features') else 3,
                model_size="small"
            ).to(device)
            model_copy.load_state_dict(base_model.state_dict())
            self.models.append(model_copy)

        print(f"Successfully created {len(self.models)} model copies on devices: {self.devices}")

    def split_pixels(self, pixel_time_series):
        """Split pixels into halves or quarters"""
        total_pixels = pixel_time_series.shape[0]

        splits = []
        if self.num_splits == 2:
            # Split into halves
            mid = total_pixels // 2

            # First half
            split1 = pixel_time_series[:mid].to(self.devices[0])
            splits.append((split1, 0, mid, self.devices[0]))

            # Second half
            split2 = pixel_time_series[mid:].to(self.devices[1])
            splits.append((split2, mid, total_pixels, self.devices[1]))

        elif self.num_splits == 4:
            # Split into quarters
            quarter = total_pixels // 4

            # First quarter
            split1 = pixel_time_series[:quarter].to(self.devices[0])
            splits.append((split1, 0, quarter, self.devices[0]))

            # Second quarter
            split2 = pixel_time_series[quarter:2 * quarter].to(self.devices[1])
            splits.append((split2, quarter, 2 * quarter, self.devices[1]))

            # Third quarter
            split3 = pixel_time_series[2 * quarter:3 * quarter].to(self.devices[2])
            splits.append((split3, 2 * quarter, 3 * quarter, self.devices[2]))

            # Fourth quarter (handles remainder)
            split4 = pixel_time_series[3 * quarter:].to(self.devices[3])
            splits.append((split4, 3 * quarter, total_pixels, self.devices[3]))

        print(f"Split {total_pixels} pixels into {len(splits)} parts:")
        for i, (_, start, end, device) in enumerate(splits):
            print(f"  Part {i}: pixels {start}-{end} ({end - start} pixels) on {device}")

        return splits

    def forward(self, pixel_time_series):
        """Forward pass with manual GPU splitting"""
        # Split data across GPUs
        splits = self.split_pixels(pixel_time_series)

        results = []
        # Process each split on its GPU
        for i, (split_data, start_idx, end_idx, device) in enumerate(splits):
            with torch.cuda.device(device):
                pred = self.models[i](split_data)
                results.append((pred, start_idx, end_idx))

        # Combine results back on main GPU
        total_pixels = pixel_time_series.shape[0]
        full_predictions = torch.zeros(total_pixels, 3, device="cuda:0")

        for pred, start_idx, end_idx in results:
            full_predictions[start_idx:end_idx] = pred.to("cuda:0")

        return full_predictions

    def parameters(self):
        """Return parameters from all models for optimizer"""
        params = []
        for model in self.models:
            params.extend(list(model.parameters()))
        return params


def print_gpu_memory_usage():
    """Print current GPU memory usage for all devices"""
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3  # GB
        cached = torch.cuda.memory_reserved(i) / 1024 ** 3  # GB
        print(f"GPU {i}: {allocated:.2f}GB allocated, {cached:.2f}GB cached")


# ===== MULTI-GPU CONFIGURATION =====
num_splits = 2  # Change to 2 or 4 based on your GPU count

# ===== SETUP PARAMETERS =====
seq_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\23.7.25\epi_gre_mrf_epi_108\epi_gre_mrf_epi.seq"
phantom_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\most_updated\numerical_brain_cropped.mat"
output_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\datasets\mrf custom dataset\epi\23.7.25\epi_gre_mrf_epi_108\run_1"

epochs = 500

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
pixel_time_series = normalized_time_series.transpose(0, 1).to("cuda")  # Shape: (num_masked_pixels, time_steps)

# Get spatial indices for reconstruction
masked_indices = torch.where(mask)
masked_rows = masked_indices[0]
masked_cols = masked_indices[1]

if plot:
    plot_phantom(phantom, save_path=os.path.join(plots_output_path, 'phantom.png'))
    display_time_series_shots(time_series_shots, flip_angles,
                              save_path=os.path.join(plots_output_path, 'time_series_shots.png'))

# ===== DEFINE NETWORK =====
from mlp import create_simple_mlp

base_model = create_simple_mlp(
    input_features=time_steps_number,  # time steps
    output_features=3,  # T1, T2, PD
    model_size="small"  # Can try larger sizes now with multi-GPU
)
base_model = base_model.to("cuda")

# Setup multi-GPU (simplified approach)
multi_gpu_model = SimpleMultiGPU(base_model, num_splits)

# ===== OPTIMIZATION SETUP =====
optimizer = torch.optim.Adam(multi_gpu_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
losses = []

# Print initial memory usage
print("=== Initial Memory Usage ===")
print_gpu_memory_usage()

# ===== MAIN TRAINING LOOP =====
print("Starting multi-GPU training...")
start_time = time.time()

for iteration in range(epochs):
    optimizer.zero_grad()

    # Forward pass: split pixels across GPUs manually
    predictions = multi_gpu_model.forward(pixel_time_series)  # Shape: (num_masked_pixels, 3)

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

    # Create phantom and simulate
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

    # Calculate normalized loss using quantiles
    relative_loss = 0
    for t in range(len(time_series_shots)):
        # Extract masked pixels for this time point
        real_t = time_series_shots[t][mask]  # Shape: (num_masked_pixels,)
        sim_t = sim_images_batch.squeeze()[t][mask]  # Shape: (num_masked_pixels,)

        # Calculate 1% and 99% quantiles for normalization (only on masked pixels)
        q1 = torch.quantile(real_t, 0.01)
        q99 = torch.quantile(real_t, 0.99)
        signal_range = q99 - q1 + 1e-8

        # Normalize both real and simulated to [0,1] range
        real_normalized = torch.clamp((real_t - q1) / signal_range, 0, 1)
        sim_normalized = torch.clamp((sim_t - q1) / signal_range, 0, 1)

        # Calculate MSE on normalized data
        single_image_loss = F.mse_loss(real_normalized, sim_normalized)
        relative_loss += single_image_loss

    # Average across time points
    per_pixel_loss = relative_loss / len(time_series_shots)
    image_loss = per_pixel_loss

    # Backward pass
    image_loss.backward()
    optimizer.step()
    scheduler.step()

    # Track loss
    losses.append(image_loss.item())

    # Progress
    print(f"Iteration {iteration}: Loss = {image_loss.item():.8f}")

    # Print memory usage periodically
    if iteration % 10 == 0:
        print_gpu_memory_usage()

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

# Print final memory usage
print("=== Final Memory Usage ===")
print_gpu_memory_usage()

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