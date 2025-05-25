# main_training_multi_param.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import MRzeroCore as mr0
import phantom_creator
import time
import sys
import os
from contextlib import redirect_stdout, redirect_stderr

# ===== SETUP PARAMETERS =====
n_TI = 4  # Number of inversion times
Nread = 32  # Readout steps
Nphase = 32  # Phase encoding steps
TI = [0.1, 0.5, 1.0, 5.0]  # Inversion times in seconds
max_iterations = 150  # Maximum training iterations
experiment_id = 'FLASH_2D_Fit'

# Plotting flag
PLOT_GROUND_TRUTH = True  # Set to False to disable plotting


def simulate_and_process_mri(obj_p, experiment_id, n_TI, Nread, Nphase, permvec, plot_results=False,
                             plot_title="MRI Simulation"):
    """
    Simulate MRI sequence and process the data

    Args:
        obj_p: Phantom object
        experiment_id: Name of the sequence file
        n_TI: Number of inversion times (or time steps)
        Nread: Readout steps
        Nphase: Phase encoding steps
        permvec: Permutation vector for phase encoding
        plot_results: Whether to plot the results
        plot_title: Title for the plots

    Returns:
        magnitude_images: Magnitude images (normalized)
        phase_images: Phase images
        kspace: K-space data
        space: Image domain data (complex)
        signal: Raw ADC signal
    """

    # Read in the sequence
    seq0 = mr0.Sequence.import_file(experiment_id + '.seq')

    # Save the original file descriptors
    original_stdout_fd = os.dup(sys.stdout.fileno())
    original_stderr_fd = os.dup(sys.stderr.fileno())

    # Open devnull
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        # Redirect stdout and stderr to devnull
        os.dup2(devnull_fd, sys.stdout.fileno())
        os.dup2(devnull_fd, sys.stderr.fileno())

        # Your MR operations
        graph = mr0.compute_graph(seq0, obj_p, 20480, 1e-3)
        signal = mr0.execute_graph(graph, seq0, obj_p, print_progress=False)

    finally:
        # Restore original stdout and stderr
        os.dup2(original_stdout_fd, sys.stdout.fileno())
        os.dup2(original_stderr_fd, sys.stderr.fileno())

        # Close file descriptors
        os.close(devnull_fd)
        os.close(original_stdout_fd)
        os.close(original_stderr_fd)

    # Process the signal - CORRECTED: Use len(TI) instead of n_TI and correct reshape order
    kspace = torch.reshape(signal, (n_TI, Nphase, Nread)).clone().permute(0, 2, 1)

    # Apply permutation vector - CORRECTED: Apply inverse permutation
    ipermvec = np.argsort(permvec)
    kspace = kspace[:, :, ipermvec]

    # FFT processing - CORRECTED: Follow exact same steps as original
    # fftshift
    spectrum = torch.fft.fftshift(kspace, dim=(1, 2))
    # FFT
    space = torch.fft.fft2(spectrum)
    # fftshift
    space = torch.fft.ifftshift(space, dim=(1, 2))

    if plot_results:
        # Calculate number of rows needed for subplots
        n_cols = 4
        n_rows_per_timestep = 2  # k-space and image plots
        total_rows = n_TI * n_rows_per_timestep

        plt.figure(figsize=(16, 4 * total_rows))
        plt.suptitle(plot_title, fontsize=16)

        plot_idx = 1

        # Plot for each time step
        for t in range(n_TI):
            # K-space magnitude
            plt.subplot(total_rows, n_cols, plot_idx)
            plt.title(f'k-space (t={t})')
            plt.imshow(np.abs(kspace[t].numpy()))
            plt.colorbar()
            plot_idx += 1

            # K-space log magnitude
            plt.subplot(total_rows, n_cols, plot_idx)
            plt.title(f'log k-space (t={t})')
            plt.imshow(np.log(np.abs(kspace[t].numpy()) + 1e-10))  # Added epsilon to avoid log(0)
            plt.colorbar()
            plot_idx += 1

            # Image magnitude
            plt.subplot(total_rows, n_cols, plot_idx)
            plt.title(f'FFT-magnitude (t={t})')
            plt.imshow(np.abs(space[t].numpy()))
            plt.colorbar()
            plot_idx += 1

            # Image phase
            plt.subplot(total_rows, n_cols, plot_idx)
            plt.title(f'FFT-phase (t={t})')
            plt.imshow(np.angle(space[t].numpy()), vmin=-np.pi, vmax=np.pi)
            plt.colorbar()
            plot_idx += 1

        plt.tight_layout()
        plt.show()

    # Create magnitude and phase images
    magnitude_images = torch.abs(space)
    phase_images = torch.angle(space)

    return magnitude_images, phase_images, kspace, space, signal


# ===== PREPARE PHANTOM =====
phantom = phantom_creator.create_phantom(Nread, Nphase)
T1_ground_truth = phantom.T1  # Store ground truth T1 map
T2_ground_truth = phantom.T2  # Store ground truth T2 map
PD_ground_truth = phantom.PD  # Store ground truth PD map
obj_p = phantom.build()  # Build phantom for simulation

# Create permutation vector
fov = 200e-3
phenc = np.arange(-Nphase // 2, Nphase // 2, 1) / fov
permvec = sorted(np.arange(len(phenc)), key=lambda x: abs(len(phenc) // 2 - x))

# ===== GENERATE INITIAL SIMULATION DATA =====
print("Generating ground truth data...")
magnitude_images_gt, phase_images_gt, kspace_gt, space_gt, signal_gt = simulate_and_process_mri(
    obj_p, experiment_id, n_TI, Nread, Nphase, permvec,
    plot_results=PLOT_GROUND_TRUTH,
    plot_title="Ground Truth Data"
)

mask = (magnitude_images_gt[0] > 8).float()

print(f"Ground truth data generated. Magnitude shape: {magnitude_images_gt.shape}, Phase shape: {phase_images_gt.shape}")

# ===== DEFINE NEURAL NETWORK FOR MULTI-PARAMETER MAPPING =====
from enhanced_t1_network import EnhancedT1MappingNet

# Create the neural network with 2*n_TI input channels (magnitude + phase)
# Now outputs 3 channels: T1, T2, PD
multi_param_net = EnhancedT1MappingNet(input_channels=2*n_TI, output_channels=3)

# For tracking progress
losses = []
t1_losses = []
t2_losses = []
pd_losses = []

# ===== OPTIMIZATION SETUP =====
optimizer = torch.optim.Adam(multi_param_net.parameters(), lr=0.001)  # Reduced learning rate
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

# Combine magnitude and phase images and add batch dimension
# Shape: [1, 2*n_TI, H, W] where first n_TI channels are magnitude, next n_TI are phase
input_images_gt_batch = torch.cat([magnitude_images_gt, phase_images_gt], dim=0).unsqueeze(0)

print(f"Input tensor shape for network: {input_images_gt_batch.shape}")

# ===== MAIN TRAINING LOOP =====
print("Starting training...")
start_time = time.time()

for iteration in range(max_iterations):
    # Zero gradients
    optimizer.zero_grad()

    # Forward pass through neural network
    param_maps = multi_param_net(input_images_gt_batch)
    
    # Extract individual parameter maps and apply proper scaling
    # Assuming EnhancedT1MappingNet outputs raw values that need scaling
    T1_predicted = torch.sigmoid(param_maps[:, 0:1, :, :]) * 5.0 * mask  # T1 range: 0-5 seconds
    T2_predicted = torch.sigmoid(param_maps[:, 1:2, :, :]) * 2.0 * mask  # T2 range: 0-2 seconds  
    PD_predicted = torch.sigmoid(param_maps[:, 2:3, :, :]) * 2.0 * mask  # PD range: 0-2 (normalized)

    # Debug parameter predictions
    # print(f"Iter {iteration} - T1 predicted range: [{T1_predicted.min().item():.3f}, {T1_predicted.max().item():.3f}]")
    # print(f"T1 ground truth range: [{T1_ground_truth.min().item():.3f}, {T1_ground_truth.max().item():.3f}]")
    # print(f"Iter {iteration} - T2 predicted range: [{T2_predicted.min().item():.3f}, {T2_predicted.max().item():.3f}]")
    # print(f"T2 ground truth range: [{T2_ground_truth.min().item():.3f}, {T2_ground_truth.max().item():.3f}]")
    # print(f"Iter {iteration} - PD predicted range: [{PD_predicted.min().item():.3f}, {PD_predicted.max().item():.3f}]")
    # print(f"PD ground truth range: [{PD_ground_truth.min().item():.3f}, {PD_ground_truth.max().item():.3f}]")

    # Create object with predicted parameter maps
    obj_p_pred = phantom_creator.create_phantom_with_custom_params(
        T1_predicted.squeeze(),
        T2_predicted.squeeze(),
        PD_predicted.squeeze(),
        Nread=Nread,
        Nphase=Nphase
    ).build()

    # USE THE SAME PROCESSING FUNCTION FOR SIMULATION
    sim_magnitude, sim_phase, sim_kspace, sim_space, sim_signal = simulate_and_process_mri(
        obj_p_pred, experiment_id, n_TI, Nread, Nphase, permvec,
        plot_results=False  # No plotting during training
    )

    # Combine simulated magnitude and phase images and add batch dimension
    sim_input_batch = torch.cat([sim_magnitude, sim_phase], dim=0).unsqueeze(0)

    # Loss calculation - comparing both magnitude and phase
    magnitude_loss = F.mse_loss(input_images_gt_batch[:, :n_TI], sim_input_batch[:, :n_TI])
    phase_loss = F.mse_loss(input_images_gt_batch[:, n_TI:], sim_input_batch[:, n_TI:])

    # Parameter-specific losses for monitoring
    t1_param_loss = F.mse_loss(T1_predicted.squeeze(), T1_ground_truth)
    t2_param_loss = F.mse_loss(T2_predicted.squeeze(), T2_ground_truth)
    pd_param_loss = F.mse_loss(PD_predicted.squeeze(), PD_ground_truth)

    # Weighted combination of losses
    total_loss = magnitude_loss + 10 * phase_loss

    # Backward pass
    total_loss.backward()

    # Add gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(multi_param_net.parameters(), max_norm=1.0)

    # Update model parameters
    optimizer.step()

    # Update learning rate scheduler
    scheduler.step()

    # Store losses
    losses.append(total_loss.item())
    t1_losses.append(t1_param_loss.item())
    t2_losses.append(t2_param_loss.item())
    pd_losses.append(pd_param_loss.item())

    # Progress update
    if iteration % 1 == 0:
        print(f"Iteration {iteration}: Total Loss = {total_loss.item():.8f} (Mag: {magnitude_loss.item():.8f}, Phase: {phase_loss.item():.8f})")
        print(f"  Parameter losses - T1: {t1_param_loss.item():.6f}, T2: {t2_param_loss.item():.6f}, PD: {pd_param_loss.item():.6f}")

    if iteration % 15 == 0:
        # Create a comprehensive plot with all parameter maps and time series images
        fig = plt.figure(figsize=(24, 20))

        # Top rows: Parameter maps comparison (Ground truth vs Predicted)
        # T1 maps
        ax1 = plt.subplot(6, 4, 1)
        im1 = ax1.imshow(T1_ground_truth.numpy(), cmap='viridis')
        ax1.set_title('Ground Truth T1 Map')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1)

        ax2 = plt.subplot(6, 4, 2)
        T1_pred_display = T1_predicted.squeeze().detach().numpy()
        im2 = ax2.imshow(T1_pred_display, cmap='viridis')
        ax2.set_title(f'Predicted T1 Map (Iter {iteration})')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)

        # T2 maps
        ax3 = plt.subplot(6, 4, 3)
        im3 = ax3.imshow(T2_ground_truth.numpy(), cmap='plasma')
        ax3.set_title('Ground Truth T2 Map')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3)

        ax4 = plt.subplot(6, 4, 4)
        T2_pred_display = T2_predicted.squeeze().detach().numpy()
        im4 = ax4.imshow(T2_pred_display, cmap='plasma')
        ax4.set_title(f'Predicted T2 Map (Iter {iteration})')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4)

        # PD maps
        ax5 = plt.subplot(6, 4, 5)
        im5 = ax5.imshow(PD_ground_truth.numpy(), cmap='hot')
        ax5.set_title('Ground Truth PD Map')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5)

        ax6 = plt.subplot(6, 4, 6)
        PD_pred_display = PD_predicted.squeeze().detach().numpy()
        im6 = ax6.imshow(PD_pred_display, cmap='hot')
        ax6.set_title(f'Predicted PD Map (Iter {iteration})')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6)

        # Third row: Real magnitude images time series
        for t in range(n_TI):
            ax = plt.subplot(6, 4, 9 + t)
            im = ax.imshow(magnitude_images_gt[t].detach().numpy(), cmap='gray')
            ax.set_title(f'Real Mag t={t}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)

        # Fourth row: Simulated magnitude images time series
        for t in range(n_TI):
            ax = plt.subplot(6, 4, 13 + t)
            im = ax.imshow(sim_magnitude[t].detach().numpy(), cmap='gray')
            ax.set_title(f'Sim Mag t={t}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)

        # Fifth row: Phase comparison (showing only first time point for space)
        ax_phase_gt = plt.subplot(6, 4, 17)
        im_phase_gt = ax_phase_gt.imshow(phase_images_gt[0].detach().numpy(), cmap='hsv', vmin=-np.pi, vmax=np.pi)
        ax_phase_gt.set_title('Real Phase t=0')
        ax_phase_gt.axis('off')
        plt.colorbar(im_phase_gt, ax=ax_phase_gt)

        ax_phase_sim = plt.subplot(6, 4, 18)
        im_phase_sim = ax_phase_sim.imshow(sim_phase[0].detach().numpy(), cmap='hsv', vmin=-np.pi, vmax=np.pi)
        ax_phase_sim.set_title('Sim Phase t=0')
        ax_phase_sim.axis('off')
        plt.colorbar(im_phase_sim, ax=ax_phase_sim)

        # Loss curves
        ax_loss = plt.subplot(6, 4, 19)
        ax_loss.semilogy(losses, label='Total Loss')
        ax_loss.set_title('Training Loss')
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)

        ax_param_loss = plt.subplot(6, 4, 20)
        ax_param_loss.semilogy(t1_losses, label='T1 Loss')
        ax_param_loss.semilogy(t2_losses, label='T2 Loss')
        ax_param_loss.semilogy(pd_losses, label='PD Loss')
        ax_param_loss.set_title('Parameter Losses')
        ax_param_loss.set_xlabel('Iteration')
        ax_param_loss.set_ylabel('Loss')
        ax_param_loss.legend()
        ax_param_loss.grid(True)

        plt.tight_layout()
        plt.show()

    # Early stopping condition
    if total_loss.item() < 1e-7:
        print(f"Converged at iteration {iteration}")
        break

# ===== FINAL RESULTS =====
total_time = time.time() - start_time
print(f"Training complete! Total time: {total_time:.2f} seconds")
print(f"Final loss: {losses[-1]:.8f}")

# Plot loss curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.semilogy(losses)
plt.title('Total Training Loss vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('MSE Loss (Log Scale)')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.semilogy(t1_losses, label='T1 Loss')
plt.semilogy(t2_losses, label='T2 Loss')
plt.semilogy(pd_losses, label='PD Loss')
plt.title('Parameter-Specific Losses')
plt.xlabel('Iteration')
plt.ylabel('MSE Loss (Log Scale)')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
# Final parameter comparison
param_maps_final = multi_param_net(input_images_gt_batch)
T1_final = param_maps_final[:, 0, :, :].squeeze().detach().numpy()
T2_final = param_maps_final[:, 1, :, :].squeeze().detach().numpy()
PD_final = param_maps_final[:, 2, :, :].squeeze().detach().numpy()

# Calculate correlation coefficients
t1_corr = np.corrcoef(T1_ground_truth.flatten(), T1_final.flatten())[0, 1]
t2_corr = np.corrcoef(T2_ground_truth.flatten(), T2_final.flatten())[0, 1]
pd_corr = np.corrcoef(PD_ground_truth.flatten(), PD_final.flatten())[0, 1]

plt.bar(['T1', 'T2', 'PD'], [t1_corr, t2_corr, pd_corr])
plt.title('Parameter Correlation Coefficients')
plt.ylabel('Correlation with Ground Truth')
plt.ylim([0, 1])
plt.grid(True)

plt.tight_layout()
plt.show()

# Save model
torch.save({
    'model_state_dict': multi_param_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_loss': losses[-1],
    'n_TI': n_TI,
    'network_type': 'MultiParameterMappingNet_T1_T2_PD',
    'input_channels': 2*n_TI,
    'output_channels': 3,
    't1_losses': t1_losses,
    't2_losses': t2_losses,
    'pd_losses': pd_losses
}, 'best_multi_parameter_mapping_model.pth')

print("Model saved to 'best_multi_parameter_mapping_model.pth'")
print(f"Final correlations - T1: {t1_corr:.4f}, T2: {t2_corr:.4f}, PD: {pd_corr:.4f}")