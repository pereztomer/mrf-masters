import os

from sympy.abc import alpha

# must be set before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch.nn.functional as F
import phantom_creator
import pypulseq as pp
from simulate_and_process import simulate_and_process_mri
from plotting_utils import *

import sys
from PIL import Image
import numpy as np
import shutil


def mask_alignment_loss(aligned_masks):
    """
    Compare all aligned masks to the first (undeformed) mask
    aligned_masks: [1, N, H, W] where first frame is undeformed reference
    """
    batch_size, num_frames = aligned_masks.shape[:2]

    # Reference mask (first frame, undeformed)
    ref_mask = aligned_masks[:, 0]  # [1, H, W]

    # Compare all other frames to reference
    mask_loss = 0
    for i in range(1, num_frames):
        mask_loss += F.mse_loss(aligned_masks[:, i], ref_mask)

    return mask_loss / (num_frames - 1)  # Average over frames


def apply_displacement_field_sequence(moving_sequence, displacement_fields):
    """
    Args:
        moving_sequence: [1, 50, H, W]
        displacement_fields: [1, 98, H, W] - network output already in [-1,1]
    """
    device = moving_sequence.device
    _, _, height, width = moving_sequence.shape

    # Identity grid [-1, 1]
    y, x = torch.meshgrid(torch.linspace(-1, 1, height, device=device),
                          torch.linspace(-1, 1, width, device=device), indexing='ij')
    base_grid = torch.stack([x, y], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    # First frame unchanged
    aligned_frames = [moving_sequence[:, 0:1]]

    # Apply displacement to frames 1-49
    for i in range(49):
        dx = displacement_fields[:, i * 2]  # [1, H, W]
        dy = displacement_fields[:, i * 2 + 1]  # [1, H, W]

        # Network outputs [-1,1] already, so use directly
        displacement = torch.stack([dx, dy], dim=-1)  # [1, H, W, 2]
        grid = base_grid - displacement

        # Warp frame
        warped = F.grid_sample(moving_sequence[:, i + 1:i + 2], grid,
                               mode='bilinear', padding_mode='border', align_corners=True)
        aligned_frames.append(warped)

    return torch.cat(aligned_frames, dim=1)


def smoothness_loss(displacement_fields):
    """
    Compute spatial smoothness regularization
    displacement_fields: [1, 98, H, W]
    """
    # Compute gradients
    dx_dx = displacement_fields[:, ::2, :, 1:] - displacement_fields[:, ::2, :, :-1]  # dx gradients
    dx_dy = displacement_fields[:, ::2, 1:, :] - displacement_fields[:, ::2, :-1, :]

    dy_dx = displacement_fields[:, 1::2, :, 1:] - displacement_fields[:, 1::2, :, :-1]  # dy gradients
    dy_dy = displacement_fields[:, 1::2, 1:, :] - displacement_fields[:, 1::2, :-1, :]

    return torch.mean(dx_dx ** 2) + torch.mean(dx_dy ** 2) + torch.mean(dy_dx ** 2) + torch.mean(dy_dy ** 2)


def edge_consistency_loss(aligned_time_series):
    batch_size, num_frames = aligned_time_series.shape[:2]
    device = aligned_time_series.device

    # Sobel edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)

    # Get edges for all frames
    reshaped = aligned_time_series.view(-1, 1, aligned_time_series.shape[-2], aligned_time_series.shape[-1])
    edges_x = F.conv2d(reshaped, sobel_x, padding=1)
    edges_y = F.conv2d(reshaped, sobel_y, padding=1)
    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2).view(aligned_time_series.shape)

    # Normalize edges to [0,1] per frame
    edges_norm = (edges - edges.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]) / \
                 (edges.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] -
                  edges.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0] + 1e-8)

    # Compare normalized edges
    ref_edges = edges_norm[:, 0]
    edge_loss = 0
    for i in range(1, num_frames):
        edge_loss += F.mse_loss(edges_norm[:, i], ref_edges)

    return edge_loss / (num_frames - 1), edges_norm


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===== SETUP PARAMETERS =====
# Device configuration
use_gpu = True
if use_gpu:
    assert torch.cuda.device_count() >= 2, f"Need at least 2 GPUs, found {torch.cuda.device_count()}"
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    torch.cuda.set_device(0)
else:
    device0 = torch.device("cpu")
    device1 = torch.device("cpu")

use_wandb = False

# ===== SETUP PARAMETERS =====
run_name = "test"
seq_path_with_reference = r"C:\Users\perez\Desktop\mrf_runs\epi_gre_54\gre_epi_54_fourier_factor_1_w_reference.seq"
seq_path_without_reference = r"C:\Users\perez\Desktop\mrf_runs\epi_gre_54\gre_epi_54_fourier_factor_1_wo_reference.seq"
phantom_path = r"C:\Users\perez\Desktop\mrf_runs\numerical_brain_cropped.mat"
output_path = fr"C:\Users\perez\Desktop\mrf_runs\epi_gre_54\{run_name}"

epochs = 2000

# ===== CREATE OUTPUT FOLDERS =====
plots_output_path = os.path.join(output_path, 'plots')
maps_output_path = os.path.join(output_path, "maps")
os.makedirs(os.path.join(maps_output_path, "T1_np_maps"), exist_ok=True)
os.makedirs(os.path.join(maps_output_path, "T2_np_maps"), exist_ok=True)
os.makedirs(os.path.join(maps_output_path, "PD_np_maps"), exist_ok=True)

os.makedirs(os.path.join(plots_output_path, "simulated_images"), exist_ok=True)
os.makedirs(os.path.join(plots_output_path, "diff_between_original_deformed_and_aligned"), exist_ok=True)
os.makedirs(os.path.join(plots_output_path, "aligned_time_series"), exist_ok=True)
os.makedirs(os.path.join(plots_output_path, "edges"), exist_ok=True)
# ===== READ SEQUENCE =====
seq_pulseq = pp.Sequence()
seq_pulseq.read(seq_path_with_reference)
Nx = int(seq_pulseq.get_definition('Nx'))
Ny = int(seq_pulseq.get_definition('Ny'))
flip_angles = seq_pulseq.get_definition('FlipAngles')
time_steps_number = len(flip_angles)
num_coils = 34
learning_rate = 0.0001
model_size = "tiny"
alpha = 1000
beta = 0
gamma = 0

if use_wandb:
    from wandb import wandb

    wandb.login(key="7573cbc6e943326835b588046bf1ee71f3f43408")
    wandb.init(
        project="mrf",
        name=run_name,
        notes="trying to commit  4ceb3a1 + old sequence file (the second version)",
        config={
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": 0,
            "Nx": Nx,
            "Ny": Ny,
            "num_coils": num_coils,
            "model_size": model_size,
            "seq_path": seq_path_with_reference,
            "phantom_path": phantom_path,
        }
    )
    # log seq file as artifact
    seq_artifact = wandb.Artifact("sequence_file_with_reference", type="sequence")
    seq_artifact.add_file(seq_path_with_reference)
    wandb.log_artifact(seq_artifact)

    seq_artifact = wandb.Artifact("sequence_file_without_reference", type="sequence")
    seq_artifact.add_file(seq_path_without_reference)
    wandb.log_artifact(seq_artifact)

    # Log phantom file as artifact
    phantom_artifact = wandb.Artifact("phantom_file", type="phantom")
    phantom_artifact.add_file(phantom_path)
    wandb.log_artifact(phantom_artifact)

# ===== PREPARE PHANTOM =====
phantom, coil_maps = phantom_creator.create_phantom(Nx, Ny, phantom_path, num_coils=num_coils)
coil_maps = coil_maps.to(device0)
obj_p = phantom.build()
T1_ground_truth = phantom.T1.squeeze().to(device0)
T2_ground_truth = phantom.T2.squeeze().to(device0)
PD_ground_truth = phantom.PD.squeeze().to(device0)

# ===== CREATE MASK AND GROUND TRUTH =====
mask = T1_ground_truth > 0
mask = mask.to(device0)

# Calculate normalization parameters
t1_mean = torch.mean(T1_ground_truth[mask])
t1_std = torch.std(T1_ground_truth[mask])
t2_mean = torch.mean(T2_ground_truth[mask])
t2_std = torch.std(T2_ground_truth[mask])
pd_mean = torch.mean(PD_ground_truth[mask])
pd_std = torch.std(PD_ground_truth[mask])

obj_p = phantom.build()  # phantom for simulation

# ===== INITIAL SIMULATION DATA =====
calibration_data, time_series_shots, grappa_weights_torch = simulate_and_process_mri(obj_p,
                                                                                     seq_path_with_reference,
                                                                                     num_coils,
                                                                                     device=device0,
                                                                                     with_reference=True)

from distortion.elasticdeform_time_series import deform_aux

padded_image_list = []
for i in range(50):
    img = time_series_shots[i].detach().cpu().numpy()
    h, w = img.shape
    pad_h, pad_w = int(h * 0.3), int(w * 0.3)
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    padded_image_list.append(padded_img)

deform = deform_aux(padded_image_list)

image_mask = padded_image_list[0] > 10
image_mask = image_mask.astype(np.uint8)
deformed_images = deform(padded_image_list, basic_sigma=10, generated_video=True, video_path=output_path)
deformed_time_series_shots = torch.tensor(np.stack(deformed_images), dtype=time_series_shots[0].dtype, device=device0)

image_masks_list = [image_mask for i in range(len(padded_image_list))]
deformed_masks = deform(image_masks_list, basic_sigma=10, generated_video=False)
deformed_masks = torch.tensor(np.stack(deformed_masks), dtype=time_series_shots[0].dtype, device=device0)

grappa_weights_torch = grappa_weights_torch.detach()

plot_phantom(phantom, save_path=os.path.join(plots_output_path, 'phantom.png'))

plot_calibration_image_vs_first_time_step(calibration_data, time_series_shots, plots_output_path)

display_time_series_shots(time_series_shots,
                          flip_angles,
                          save_path=os.path.join(plots_output_path, 'time_series_shots.png'))

display_time_series_shots(deformed_time_series_shots,
                          flip_angles,
                          save_path=os.path.join(plots_output_path, 'deformed_time_series_shots.png'))

# ===== DEFINE NETWORK =====
from pipeline_gre.models.conv_unet import create_3d_unet_mri

maps_extraction_model = create_3d_unet_mri(
    time_steps=len(time_series_shots),  # Your time dimension
    input_shape=(Nx, Nx),  # Your spatial dimensions
    out_channels=3,  # T1, T2, PD outputs
    model_size=model_size  # Options: "tiny", "small", "medium", "large", "huge"
)

maps_registration_model = create_3d_unet_mri(
    time_steps=len(time_series_shots),  # Your time dimension
    input_shape=(Nx, Nx),  # Your spatial dimensions
    out_channels=(len(time_series_shots) - 1) * 2,  # T1, T2, PD outputs
    model_size=model_size,  # Options: "tiny", "small", "medium", "large", "huge"
    registration_map=True  # This adds tanh activation
)

maps_extraction_model = maps_extraction_model.to(device0)
maps_registration_model = maps_registration_model.to(device1)
# ===== OPTIMIZATION SETUP =====
maps_extraction_optimizer = torch.optim.Adam(maps_extraction_model.parameters(), lr=0.0001, weight_decay=0)
maps_extraction_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(maps_extraction_optimizer, T_max=epochs,
                                                                       eta_min=1e-6)

# Add this after your existing optimizer setup:
registration_optimizer = torch.optim.Adam(maps_registration_model.parameters(), lr=0.0001, weight_decay=0)
registration_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(registration_optimizer, T_max=epochs, eta_min=1e-6)

best_loos = 1000000000000
best_t1_loss = 1000000000000
best_t2_loss = 1000000000000
best_pd_loss = 1000000000000

# ===== MAIN TRAINING LOOP =====
losses = []
t1_losses = []
t2_losses = []
pd_losses = []

# ===== normelize images ========
reshaped = time_series_shots.view(50, -1)  # [50, 1296]

# Calculate mean and std along the spatial dimensions (dim=1)
mean = reshaped.mean(dim=1, keepdim=True)  # [50, 1]
std = reshaped.std(dim=1, keepdim=True)  # [50, 1]

# Normalize
normalized_reshaped = (deformed_time_series_shots.view(50, -1) - mean) / (
        std + 1e-8)  # Add small epsilon to avoid div by zero

# Reshape back to original dimensions
time_series_shots_normalized = normalized_reshaped.view(50, Nx + 2 * pad_h, Nx + 2 * pad_h)

# ===== ADD BATCH DIMENSION =====
time_series_shots_normalized = time_series_shots_normalized.unsqueeze(0)  # shape: [1, n_TI, H, W]

# ===== MAIN TRAINING LOOP =====
for iteration in range(epochs):
    maps_extraction_optimizer.zero_grad()
    registration_optimizer.zero_grad()

    # Forward
    parameters_heat_map_predictions = maps_extraction_model(time_series_shots_normalized.to(device0))
    parameters_heat_map_predictions = parameters_heat_map_predictions[..., pad_h:-pad_h, pad_w:-pad_w]

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

    current_t1_loss = F.mse_loss(t1_predicted, T1_ground_truth)
    current_t2_loss = F.mse_loss(t2_predicted, T2_ground_truth)
    current_pd_loss = F.mse_loss(pd_predicted, PD_ground_truth)

    # Create phantom with predicted T1,T2 and PD
    obj_p_pred = phantom_creator.create_phantom_with_custom_parameters(
        T1_map=t1_predicted,
        T2_map=t2_predicted,
        PD_map=pd_predicted,
        Nread=Nx,
        Nphase=Ny,
        phantom_path=phantom_path,
        coil_maps=coil_maps,
        device=device0
    )

    obj_p_pred = obj_p_pred.build()
    # Simulate images with predicted T1
    sim_images = simulate_and_process_mri(obj_p_pred,
                                          seq_path_without_reference,
                                          num_coils,
                                          device=device0,
                                          with_reference=False,
                                          grappa_weights_torch=grappa_weights_torch)

    sim_images = F.pad(sim_images, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
    sim_images = sim_images.unsqueeze(0)

    time_series_shots_normalized = time_series_shots_normalized.to(device1)
    registration_maps = maps_registration_model(time_series_shots_normalized)

    # Print displacement statistics before and after scaling
    print(f"Raw displacement range: [{registration_maps.min():.3f}, {registration_maps.max():.3f}]")

    original_size = Nx  # 54 in your case
    padded_size = Nx + 2 * pad_h  # your current padded size
    scale_factor = original_size / padded_size

    registration_maps = registration_maps * scale_factor

    # test_disp = torch.zeros(1, 98, padded_size, padded_size)
    # test_disp[:, 0::2, :, :] = 0.1  # dx = 0.1 for all frames
    # aligned_time_series = apply_displacement_field_sequence(time_series_shots_normalized, test_disp.to(device1))

    aligned_time_series = apply_displacement_field_sequence(moving_sequence=time_series_shots_normalized,
                                                            displacement_fields=registration_maps)

    aligned_masks = apply_displacement_field_sequence(moving_sequence=deformed_masks.unsqueeze(0).to(device1),
                                                      displacement_fields=registration_maps)

    diff = aligned_time_series - time_series_shots_normalized

    plot_k_images(time_series=diff.detach().cpu().numpy(),
                  k=10,
                  save_path=os.path.join(plots_output_path, "diff_between_original_deformed_and_aligned",
                                         f"diff_between_original_deformed_and_aligned{iteration}.png"),
                  title=f"diff_between_original_deformed_and_aligned_{iteration}"
                  )
    plot_k_images(time_series=sim_images.detach().cpu().numpy(),
                  k=10,
                  save_path=os.path.join(plots_output_path, "simulated_images",
                                         f"simulated_images_iteration_{iteration}.png"),
                  title=f"Simulated Images_{iteration}")
    plot_k_images(time_series=aligned_time_series.detach().cpu().numpy(),
                  k=10,
                  save_path=os.path.join(plots_output_path, "aligned_time_series",
                                         f"aligned_time_series_iteration_{iteration}.png"),
                  title=f"aligned_time_series_iteration_{iteration}")

    # Compute loss
    image_loss = F.mse_loss(aligned_time_series.to(device0), sim_images)
    smooth_loss = smoothness_loss(registration_maps)
    registration_loss, edges_norm = edge_consistency_loss(aligned_time_series.squeeze())
    alignment_mask_loss = mask_alignment_loss(aligned_masks)
    total_loss = image_loss + alpha * alignment_mask_loss.to(device0) + beta * smooth_loss.to(device0) + gamma * registration_loss.to(device0)



    print(f"aligment loss: {alignment_mask_loss}")
    plot_k_images(time_series=edges_norm.unsqueeze(0).detach().cpu().numpy(),
                  k=10,
                  save_path=os.path.join(plots_output_path, "edges",
                                         f"edges_iteration_{iteration}.png"),
                  title=f"edges_iteration_{iteration}")

    losses.append(image_loss.item())
    t1_losses.append(current_t1_loss.item())
    t2_losses.append(current_t2_loss.item())
    pd_losses.append(current_pd_loss.item())

    log_dict = {
        "iteration": iteration,
        "total_loss": total_loss.item(),
        "image_loss": image_loss.item(),
        "smooth_loss": smooth_loss.item(),
        "t1_loss": current_t1_loss.item(),
        "t2_loss": current_t2_loss.item(),
        "pd_loss": current_pd_loss.item(),
        "learning_rate": maps_extraction_scheduler.get_last_lr()[0],
    }

    # Backward
    total_loss.backward()

    # optimize map extraction network
    maps_extraction_optimizer.step()
    maps_extraction_scheduler.step()

    # optimize registration network
    registration_optimizer.step()
    registration_scheduler.step()

    if (current_t1_loss < best_t1_loss or
            current_t2_loss < best_t2_loss or
            current_pd_loss < best_pd_loss or
            image_loss < best_loos):
        plot_training_results(iteration, epochs, losses, T1_ground_truth, T2_ground_truth, PD_ground_truth,
                              t1_predicted, t2_predicted, pd_predicted, time_series_shots, sim_images,
                              plots_output_path, t1_losses, t2_losses, pd_losses)

        if use_wandb:
            log_dict.update({
                "plots/training_results/training_results": wandb.Image(
                    f"{plots_output_path}/training_results/iter_{iteration:04d}.png"),
                "plots/loss_curves/loss_curves": wandb.Image(f"{plots_output_path}/loss_curves/loss_curve.png"),
            })

    if iteration % 25 == 0:
        create_video_from_training_results(f"{plots_output_path}/training_results",
                                           os.path.join(plots_output_path, "progression_video.mp4"), fps=8)
    if current_t1_loss < best_t1_loss:
        best_t1_loss = current_t1_loss
    if current_t2_loss < best_t2_loss:
        best_t2_loss = current_t2_loss
    if current_pd_loss < best_pd_loss:
        best_pd_loss = current_pd_loss
    if image_loss < best_loos:
        best_loos = image_loss
        # Save locally
        np.save(os.path.join(maps_output_path, "T1_np_maps", f'T1_best_iter_{iteration:04d}.npy'),
                t1_predicted.detach().cpu().numpy())
        np.save(os.path.join(maps_output_path, "T2_np_maps", f'T2_best_iter_{iteration:04d}.npy'),
                t2_predicted.detach().cpu().numpy())
        np.save(os.path.join(maps_output_path, "PD_np_maps", f'PD_best_iter_{iteration:04d}.npy'),
                pd_predicted.detach().cpu().numpy())

        if use_wandb:

            maps_data = [
                (T1_ground_truth, t1_predicted, 'T1'),
                (T2_ground_truth, t2_predicted, 'T2'),
                (PD_ground_truth, pd_predicted, 'PD')
            ]

            for gt, pred, name in maps_data:
                # Get numpy arrays
                gt_np = gt.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()

                # Set min to 0, max from ground truth only
                vmin = 0
                vmax = gt_np.max()

                # Clip prediction to the ground truth range
                pred_clipped = np.clip(pred_np, vmin, vmax)

                # Apply scaling/normalization
                pred_scaled = (pred_clipped - vmin) / (vmax - vmin)

                # Resize to 1024x1024 using PIL
                pred_pil = Image.fromarray((pred_scaled * 255).astype(np.uint8))
                pred_resized_pil = pred_pil.resize((728, 728), Image.BILINEAR)
                pred_resized = np.array(pred_resized_pil) / 255.0

                log_dict.update({
                    f"maps/{name}_maps/{name}_Pred": wandb.Image(pred_resized, caption=f"Pred {name}")
                })

            wandb.log(log_dict)

        # Progress
        print(f"Iteration {iteration}: Loss = {image_loss.item():.8f}")

        # Early stopping
        if image_loss.item() < 1e-7:
            print(f"Converged at iteration {iteration}")
            break

wandb.finish()
