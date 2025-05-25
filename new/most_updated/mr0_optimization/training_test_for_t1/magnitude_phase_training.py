# main_training.py

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

# ===== DEFINE NEURAL NETWORK =====
from t1_resnet_network import T1MappingNet
from enhanced_t1_network import EnhancedT1MappingNet

# Create the neural network with 2*n_TI input channels (magnitude + phase)
# T1_net = T1MappingNet(input_channels=2*n_TI, output_channels=1)
T1_net = EnhancedT1MappingNet(input_channels=2*n_TI, output_channels=1)

# For tracking progress
losses = []

# ===== OPTIMIZATION SETUP =====
optimizer = torch.optim.Adam(T1_net.parameters(), lr=0.0001)  # Reduced learning rate
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
    T1_predicted = T1_net(input_images_gt_batch)
    T1_predicted = (T1_predicted.squeeze() * mask).unsqueeze(0).unsqueeze(0)

    # Debug T1 predictions
    print(f"Iter {iteration} - T1 predicted range: [{T1_predicted.min().item():.3f}, {T1_predicted.max().item():.3f}]")
    print(f"T1 ground truth range: [{T1_ground_truth.min().item():.3f}, {T1_ground_truth.max().item():.3f}]")

    # Create object with predicted T1 map
    obj_p_pred = phantom_creator.create_phantom_with_custom_T1(
        T1_predicted.squeeze(),
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

    # Weighted combination of losses (you can adjust these weights)
    total_loss = magnitude_loss + 10 * phase_loss  # Phase loss weighted lower

    # Alternative: You could also use a single combined loss
    # total_loss = F.mse_loss(input_images_gt_batch, sim_input_batch)

    # Backward pass
    total_loss.backward()

    # Add gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(T1_net.parameters(), max_norm=1.0)

    # Update model parameters
    optimizer.step()

    # Update learning rate scheduler
    scheduler.step()

    # Store loss
    losses.append(total_loss.item())

    # Progress update
    if iteration % 1 == 0:
        print(f"Iteration {iteration}: Total Loss = {total_loss.item():.8f} (Mag: {magnitude_loss.item():.8f}, Phase: {phase_loss.item():.8f})")

    if iteration % 15 == 0:
        # Create a comprehensive plot with T1 maps and time series images
        fig = plt.figure(figsize=(20, 16))

        # Top row: T1 maps comparison
        # Ground truth T1 map
        ax1 = plt.subplot(4, 4, 1)
        im1 = ax1.imshow(T1_ground_truth.numpy(), cmap='viridis')
        ax1.set_title('Ground Truth T1 Map')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1)

        # Predicted T1 map
        ax2 = plt.subplot(4, 4, 2)
        T1_pred_display = T1_predicted.squeeze().detach().numpy()
        im2 = ax2.imshow(T1_pred_display, cmap='viridis')
        ax2.set_title(f'Predicted T1 Map (Iter {iteration})')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)

        # Second row: Real magnitude images time series
        for t in range(n_TI):
            ax = plt.subplot(4, 4, 5 + t)
            im = ax.imshow(magnitude_images_gt[t].detach().numpy(), cmap='gray')
            ax.set_title(f'Real Mag t={t}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)

        # Third row: Simulated magnitude images time series
        for t in range(n_TI):
            ax = plt.subplot(4, 4, 9 + t)
            im = ax.imshow(sim_magnitude[t].detach().numpy(), cmap='gray')
            ax.set_title(f'Sim Mag t={t}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)

        # Fourth row: Phase comparison (showing only first time point for space)
        ax_phase_gt = plt.subplot(4, 4, 13)
        im_phase_gt = ax_phase_gt.imshow(phase_images_gt[0].detach().numpy(), cmap='hsv', vmin=-np.pi, vmax=np.pi)
        ax_phase_gt.set_title('Real Phase t=0')
        ax_phase_gt.axis('off')
        plt.colorbar(im_phase_gt, ax=ax_phase_gt)

        ax_phase_sim = plt.subplot(4, 4, 14)
        im_phase_sim = ax_phase_sim.imshow(sim_phase[0].detach().numpy(), cmap='hsv', vmin=-np.pi, vmax=np.pi)
        ax_phase_sim.set_title('Sim Phase t=0')
        ax_phase_sim.axis('off')
        plt.colorbar(im_phase_sim, ax=ax_phase_sim)

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

# Plot only the loss curve
plt.figure(figsize=(10, 6))
plt.semilogy(losses)
plt.title('Training Loss vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('MSE Loss (Log Scale)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Save model
torch.save({
    'model_state_dict': T1_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_loss': losses[-1],
    'n_TI': n_TI,
    'network_type': 'T1MappingNet_ResNet18_MagPhase',
    'input_channels': 2*n_TI
}, 'best_t1_mapping_model_magphase.pth')

print("Model saved to 'best_t1_mapping_model_magphase.pth'")