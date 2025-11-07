
import torch.nn.functional as F
import phantom_creator
import pypulseq as pp
from simulate_and_process import simulate_and_process_mri
from plotting_utils import *
import os

torch.cuda.set_device(0)
use_wandb = False

# ===== SETUP PARAMETERS =====
# run_name = "72X72_gre_epi_on_server_run_2"
# seq_path = r"/home/tomer.perez/workspace/runs/gre_epi_72_22.8.25/gre_epi_72_fourier_factor_1.seq"
# phantom_path = r"/home/tomer.perez/workspace/data/numerical_brain_cropped.mat"
# output_path = fr"/home/tomer.perez/workspace/runs/gre_epi_72_22.8.25/{run_name}"

run_name = "test"
seq_path = r"C:\Users\perez\Desktop\mrf_runs\gre_epi_36_from_server_2\epi_gre_mrf_epi.seq"
phantom_path =r"C:\Users\perez\Desktop\mrf_runs\numerical_brain_cropped.mat"
output_path = fr"C:\Users\perez\Desktop\mrf_runs\gre_epi_36_from_server_2/{run_name}"

epochs = 10000

# ===== CREATE OUTPUT FOLDERS =====
plots_output_path = os.path.join(output_path, 'plots')
maps_output_path = os.path.join(output_path, "maps")
os.makedirs(maps_output_path, exist_ok=True)
os.makedirs(plots_output_path, exist_ok=True)

# ===== READ SEQUENCE =====
seq_pulseq = pp.Sequence()
seq_pulseq.read(seq_path)
Nx = int(seq_pulseq.get_definition('Nx'))
Ny = int(seq_pulseq.get_definition('Ny'))
flip_angles = seq_pulseq.get_definition('FlipAngles')
time_steps_number = len(flip_angles)
num_coils = 34

model_size = "tiny"
learning_rate = 0.0001
if use_wandb:
    import wandb
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
            "seq_path": seq_path,
            "phantom_path": phantom_path,
        }
    )
    # log seq file as artifact
    seq_artifact = wandb.Artifact("sequence_file", type="sequence")
    seq_artifact.add_file(seq_path)
    wandb.log_artifact(seq_artifact)
    # Log phantom file as artifact
    phantom_artifact = wandb.Artifact("phantom_file", type="phantom")
    phantom_artifact.add_file(phantom_path)
    wandb.log_artifact(phantom_artifact)

# ===== PREPARE PHANTOM =====
phantom, coil_maps = phantom_creator.create_phantom(Nx, Ny, phantom_path, num_coils=num_coils)
coil_maps = coil_maps.cuda()
obj_p = phantom.build()

# ===== INITIAL SIMULATION DATA =====
calibration_data, time_series_shots, grappa_weights_torch = simulate_and_process_mri(obj_p, seq_path, num_coils)
grappa_weights_torch = grappa_weights_torch.detach()

T1_ground_truth = phantom.T1.squeeze().cuda()
T2_ground_truth = phantom.T2.squeeze().cuda()
PD_ground_truth = phantom.PD.squeeze().cuda()

# ===== CREATE MASK AND GROUND TRUTH =====
mask = T1_ground_truth > 0
mask = mask.cuda()

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
pixel_time_series = normalized_time_series.transpose(0, 1).cuda() # Shape: (665, 50)

# Get spatial indices for reconstruction
masked_indices = torch.where(mask)
masked_rows = masked_indices[0]
masked_cols = masked_indices[1]

plot_phantom(phantom, save_path=os.path.join(plots_output_path, 'phantom.png'))

plot_calibration_image_vs_first_time_step(calibration_data, time_series_shots, plots_output_path)

display_time_series_shots(time_series_shots, flip_angles,
                          save_path=os.path.join(plots_output_path, 'time_series_shots.png'))

# ===== DEFINE NETWORK =====
from pipeline_gre.models.mlp import create_simple_mlp

model = create_simple_mlp(
    input_features=time_steps_number,  # 50 time steps
    output_features=3,  # T1, T2, PD
    model_size=model_size
)
model = model.cuda()

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
for iteration in range(epochs):
    optimizer.zero_grad()

    # Forward pass: all pixels at once
    predictions = model(pixel_time_series)

    # Reconstruct spatial maps
    t1_predicted = torch.zeros_like(T1_ground_truth).cuda()
    t2_predicted = torch.zeros_like(T2_ground_truth).cuda()
    pd_predicted = torch.zeros_like(PD_ground_truth).cuda()

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

    # Calculate masked loss (only for brain regions)
    mask_expanded = mask.unsqueeze(0).expand_as(time_series_shots)  # Shape: (50, Nx, Nx)

    image_loss = F.mse_loss(time_series_shots[mask_expanded], sim_images_batch.squeeze()[mask_expanded])

    losses.append(image_loss.item())
    t1_losses.append(current_t1_loss.item())
    t2_losses.append(current_t2_loss.item())
    pd_losses.append(current_pd_loss.item())

    log_dict = {
        "iteration": iteration,
        "total_loss": image_loss.item(),
        "t1_loss": current_t1_loss.item(),
        "t2_loss": current_t2_loss.item(),
        "pd_loss": current_pd_loss.item(),
        "learning_rate": scheduler.get_last_lr()[0],
    }

    # Backward pass
    image_loss.backward()

    optimizer.step()
    scheduler.step()
    # Progress
    print(f"Iteration {iteration}: Loss = {image_loss.item():.8f}")

    # Plot results
    if (current_t1_loss < best_t1_loss or
            current_t2_loss < best_t2_loss or
            current_pd_loss < best_pd_loss or
            image_loss < best_loos):
        plot_training_results(iteration, epochs, losses, T1_ground_truth, T2_ground_truth, PD_ground_truth,
                              t1_predicted, t2_predicted, pd_predicted, time_series_shots, sim_images_batch,
                              plots_output_path, t1_losses, t2_losses, pd_losses)

        if use_wandb:
            log_dict.update({
                "training_results": wandb.Image(f"{plots_output_path}/iterations/iter_{iteration:04d}.png"),
                "loss_curves": wandb.Image(f"{plots_output_path}/loss_curve.png"),
            })

    if current_t1_loss < best_t1_loss:
        best_t1_loss = current_t1_loss
    if current_t2_loss < best_t2_loss:
        best_t2_loss = current_t2_loss
    if current_pd_loss < best_pd_loss:
        best_pd_loss = current_pd_loss
    if image_loss < best_loos:
        best_loos = image_loss
        # Save locally
        np.save(os.path.join(maps_output_path, f'T1_best_iter_{iteration:04d}.npy'),
                t1_predicted.detach().cpu().numpy())
        np.save(os.path.join(maps_output_path, f'T2_best_iter_{iteration:04d}.npy'),
                t2_predicted.detach().cpu().numpy())
        np.save(os.path.join(maps_output_path, f'PD_best_iter_{iteration:04d}.npy'),
                pd_predicted.detach().cpu().numpy())

        if use_wandb:
            log_dict.update({
                "T1_best_map": wandb.Image(t1_predicted.detach().cpu().numpy()),
                "T2_best_map": wandb.Image(t2_predicted.detach().cpu().numpy()),
                "PD_best_map": wandb.Image(pd_predicted.detach().cpu().numpy()),
            })

    if use_wandb:
        wandb.log(log_dict)
    # Early stopping
    if image_loss.item() < 1e-7:
        print(f"Converged at iteration {iteration}")
        break

wandb.finish()
