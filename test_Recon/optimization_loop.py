import os

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===== SETUP PARAMETERS =====
# Device configuration
device_config = "cuda:0"  # Options: "cpu", "cuda:0", "cuda:1", etc.
# device_config = "cpu"  # Options: "cpu", "cuda:0", "cuda:1", etc.

if device_config == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device(device_config)
    device_number = int(device_config.split(':')[1])
    torch.cuda.set_device(device_number)

use_wandb = False

# ===== SETUP PARAMETERS =====
# run_name = "72X72_conv_run_3"
# seq_path_with_reference = r"/home/tomer.perez/workspace/runs/gre_epi_72_25.8.25_proper_delay/gre_epi_72_fourier_factor_1_w_reference.seq"
# seq_path_without_reference = r"/home/tomer.perez/workspace/runs/gre_epi_72_25.8.25_proper_delay/gre_epi_72_fourier_factor_1_wo_reference.seq"
# phantom_path = r"/home/tomer.perez/workspace/data/numerical_brain_cropped.mat"
# output_path = fr"/home/tomer.perez/workspace/runs/gre_epi_72_25.8.25_proper_delay/{run_name}"
#
# run_name = "84X84_conv_run_5"
# seq_path_with_reference = r"/home/tomer.perez/workspace/runs/gre_epi_84/gre_epi_84_fourier_factor_1_w_reference.seq"
# seq_path_without_reference = r"/home/tomer.perez/workspace/runs/gre_epi_84/gre_epi_84_fourier_factor_1_wo_reference.seq"
# phantom_path = r"/home/tomer.perez/workspace/data/numerical_brain_cropped.mat"
# output_path = fr"/home/tomer.perez/workspace/runs/gre_epi_84/{run_name}"

# run_name = "132X132_cpu_conv_run_2"
# seq_path_with_reference = r"/home/tomer.perez/workspace/runs/gre_epi_132/gre_epi_132_fourier_factor_1_w_reference.seq"
# seq_path_without_reference = r"/home/tomer.perez/workspace/runs/gre_epi_132/gre_epi_132_fourier_factor_1_wo_reference.seq"
# phantom_path = r"/home/tomer.perez/workspace/data/numerical_brain_cropped.mat"
# output_path = fr"/home/tomer.perez/workspace/runs/gre_epi_132/{run_name}"

run_name = "test"
seq_path_with_reference = r"C:\Users\perez\Desktop\mrf-motion\sequence_generation\gre_epi_36_fourier_factor_1_w_reference.seq"
seq_path_without_reference = r"C:\Users\perez\Desktop\mrf-motion\sequence_generation\gre_epi_36_fourier_factor_1_wo_reference.seq"
phantom_path = r"C:\Users\perez\Desktop\mrf_runs\numerical_brain_cropped.mat"
output_path = fr"C:\Users\perez\Desktop\mrf_runs\{run_name}"

epochs = 2000

# ===== CREATE OUTPUT FOLDERS =====
plots_output_path = os.path.join(output_path, 'plots')
maps_output_path = os.path.join(output_path, "maps")
os.makedirs(maps_output_path, exist_ok=True)
os.makedirs(plots_output_path, exist_ok=True)
os.makedirs(os.path.join(maps_output_path, "T1_np_maps"), exist_ok=True)
os.makedirs(os.path.join(maps_output_path, "T2_np_maps"), exist_ok=True)
os.makedirs(os.path.join(maps_output_path, "PD_np_maps"), exist_ok=True)
os.makedirs(os.path.join(maps_output_path, "B1_np_maps"), exist_ok=True)

# ===== READ SEQUENCE =====
seq_pulseq = pp.Sequence()
seq_pulseq.read(seq_path_with_reference)
Nx = int(seq_pulseq.get_definition('Nx'))
Ny = int(seq_pulseq.get_definition('Ny'))
flip_angles = seq_pulseq.get_definition('FlipAngles')
time_steps_number = len(flip_angles)
num_coils = 34
learning_rate = 0.0001
model_size = "large"
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
coil_maps = coil_maps.to(device)
obj_p = phantom.build()
T1_ground_truth = phantom.T1.squeeze().to(device)
T2_ground_truth = phantom.T2.squeeze().to(device)
PD_ground_truth = phantom.PD.squeeze().to(device)
B1_ground_truth = phantom.B1.squeeze().abs().to(device)

# ===== CREATE MASK AND GROUND TRUTH =====
mask = T1_ground_truth > 0
mask = mask.to(device)

# Calculate normalization parameters
t1_mean = torch.mean(T1_ground_truth[mask])
t1_std = torch.std(T1_ground_truth[mask])
t2_mean = torch.mean(T2_ground_truth[mask])
t2_std = torch.std(T2_ground_truth[mask])
pd_mean = torch.mean(PD_ground_truth[mask])
pd_std = torch.std(PD_ground_truth[mask])
b1_mean = torch.mean(B1_ground_truth[mask])
b1_std = torch.std(B1_ground_truth[mask])

obj_p = phantom.build()  # phantom for simulation

# ===== INITIAL SIMULATION DATA =====
calibration_data, time_series_shots, grappa_weights_torch = simulate_and_process_mri(obj_p,
                                                                                     seq_path_with_reference,
                                                                                     num_coils,
                                                                                     device=device,
                                                                                     with_reference=True)
grappa_weights_torch = grappa_weights_torch.detach()

plot_phantom(phantom, save_path=os.path.join(plots_output_path, 'phantom.png'))

plot_calibration_image_vs_first_time_step(calibration_data, time_series_shots, plots_output_path)

display_time_series_shots(time_series_shots, flip_angles,
                          save_path=os.path.join(plots_output_path, 'time_series_shots.png'))

# ===== DEFINE NETWORK =====
from pipeline_gre.models.conv_unet import create_3d_unet_mri

model = create_3d_unet_mri(
    time_steps=len(time_series_shots),  # Your time dimension
    input_shape=(Nx, Nx),  # Your spatial dimensions
    out_channels=4,  # T1, T2, PD, B1 outputs
    model_size=model_size  # Options: "tiny", "small", "medium", "large", "huge"
)

model = model.to(device)
# ===== OPTIMIZATION SETUP =====
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

best_loos = 1000000000000
best_t1_loss = 1000000000000
best_t2_loss = 1000000000000
best_pd_loss = 1000000000000

# ===== MAIN TRAINING LOOP =====
losses = []
t1_losses = []
t2_losses = []
pd_losses = []
b1_losses = []

# ===== normelize images ========
reshaped = time_series_shots.view(50, -1)  # [50, 1296]

# Calculate mean and std along the spatial dimensions (dim=1)
mean = reshaped.mean(dim=1, keepdim=True)  # [50, 1]
std = reshaped.std(dim=1, keepdim=True)  # [50, 1]

# Normalize
normalized_reshaped = (reshaped - mean) / (std + 1e-8)  # Add small epsilon to avoid div by zero

# Reshape back to original dimensions
time_series_shots_normalized = normalized_reshaped.view(50, Nx, Nx)

# ===== ADD BATCH DIMENSION =====
time_series_shots_normalized = time_series_shots_normalized.unsqueeze(0)  # shape: [1, n_TI, H, W]

# ===== MAIN TRAINING LOOP =====
for iteration in range(epochs):
    optimizer.zero_grad()

    # Forward
    parameters_heat_map_predictions = model(time_series_shots_normalized)

    t1_predicted = parameters_heat_map_predictions.squeeze()[0]
    t2_predicted = parameters_heat_map_predictions.squeeze()[1]
    pd_predicted = parameters_heat_map_predictions.squeeze()[2]
    b1_predicted = parameters_heat_map_predictions.squeeze()[3]

    # denormalizing for mri simulator
    t1_predicted = t1_predicted * t1_std + t1_mean
    t2_predicted = t2_predicted * t2_std + t2_mean
    pd_predicted = pd_predicted * pd_std + pd_mean
    b1_predicted = b1_predicted * b1_std + b1_mean

    t1_predicted = t1_predicted * mask.float()
    t2_predicted = t2_predicted * mask.float()
    pd_predicted = pd_predicted * mask.float()
    b1_predicted = b1_predicted * b1_std

    current_t1_loss = F.mse_loss(t1_predicted, T1_ground_truth)
    current_t2_loss = F.mse_loss(t2_predicted, T2_ground_truth)
    current_pd_loss = F.mse_loss(pd_predicted, PD_ground_truth)
    current_b1_loss = F.mse_loss(b1_predicted, B1_ground_truth)

    # Create phantom with predicted T1,T2 and PD
    obj_p_pred = phantom_creator.create_phantom_with_custom_parameters(
        T1_map=t1_predicted,
        T2_map=t2_predicted,
        PD_map=pd_predicted,
        B1_map=b1_predicted,
        Nread=Nx,
        Nphase=Ny,
        phantom_path=phantom_path,
        coil_maps=coil_maps,
        device=device
    )

    obj_p_pred = obj_p_pred.build()
    # Simulate images with predicted T1
    sim_images = simulate_and_process_mri(obj_p_pred,
                                          seq_path_without_reference,
                                          num_coils,
                                          device=device,
                                          with_reference=False,
                                          grappa_weights_torch=grappa_weights_torch)
    sim_images_batch = sim_images.unsqueeze(0)

    # Compute loss
    image_loss = F.mse_loss(time_series_shots.unsqueeze(0), sim_images_batch)

    losses.append(image_loss.item())
    t1_losses.append(current_t1_loss.item())
    t2_losses.append(current_t2_loss.item())
    pd_losses.append(current_pd_loss.item())
    b1_losses.append(current_b1_loss.item())

    log_dict = {
        "iteration": iteration,
        "total_loss": image_loss.item(),
        "t1_loss": current_t1_loss.item(),
        "t2_loss": current_t2_loss.item(),
        "pd_loss": current_pd_loss.item(),
        "b1_loss": current_b1_loss.item(),
        "learning_rate": scheduler.get_last_lr()[0],
    }

    # Backward
    image_loss.backward()
    optimizer.step()
    scheduler.step()

    # plotting and saving #
    plot_training_results(iteration, epochs, losses,
                          T1_ground_truth, T2_ground_truth, PD_ground_truth,B1_ground_truth,
                          t1_predicted, t2_predicted, pd_predicted,b1_predicted,
                          time_series_shots, sim_images_batch,
                          plots_output_path,
                          t1_losses, t2_losses, pd_losses, b1_losses)


    if iteration % 25 == 0:
        create_video_from_training_results(f"{plots_output_path}/training_results",
                                           os.path.join(plots_output_path, "progression_video.mp4"), fps=8)

    # Save locally
    np.save(os.path.join(maps_output_path, "T1_np_maps", f'T1_best_iter_{iteration:04d}.npy'),
            t1_predicted.detach().cpu().numpy())
    np.save(os.path.join(maps_output_path, "T2_np_maps", f'T2_best_iter_{iteration:04d}.npy'),
            t2_predicted.detach().cpu().numpy())
    np.save(os.path.join(maps_output_path, "PD_np_maps", f'PD_best_iter_{iteration:04d}.npy'),
            pd_predicted.detach().cpu().numpy())
    np.save(os.path.join(maps_output_path, "B1_np_maps", f'B1_best_iter_{iteration:04d}.npy'),
            b1_predicted.detach().cpu().numpy())

    if use_wandb:
        log_dict.update({
            "plots/training_results/training_results": wandb.Image(
                f"{plots_output_path}/training_results/iter_{iteration:04d}.png"),
            "plots/loss_curves/loss_curves": wandb.Image(f"{plots_output_path}/loss_curves/loss_curve.png"),
        })

        maps_data = [
            (T1_ground_truth, t1_predicted, 'T1'),
            (T2_ground_truth, t2_predicted, 'T2'),
            (PD_ground_truth, pd_predicted, 'PD'),
            (B1_ground_truth, b1_predicted, 'B1'),
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
