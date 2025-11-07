import numpy as np

# import matplotlib
#
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import os
import cv2

def plot_phantom(phantom, num_sens_to_show=4, save_path=None):
    total_plots = 4 + num_sens_to_show
    ncols = 4
    nrows = (total_plots + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axs = axs.flatten()

    pd_np = phantom.PD.squeeze().detach().cpu().numpy()
    im = axs[0].imshow(np.abs(pd_np).T)
    axs[0].set_title('Proton Density (PD)')
    plt.colorbar(im, ax=axs[0])

    t1_np = phantom.T1.squeeze().detach().cpu().numpy()
    im = axs[1].imshow(t1_np.T)
    axs[1].set_title('T1 Relaxation Time (s)')
    plt.colorbar(im, ax=axs[1])

    t2_np = phantom.T2.squeeze().detach().cpu().numpy()
    im = axs[2].imshow(t2_np.T)
    axs[2].set_title('T2 Relaxation Time (s)')
    plt.colorbar(im, ax=axs[2])

    b1_np = phantom.B1.squeeze().detach().abs().cpu().numpy()
    im = axs[3].imshow(b1_np.T)
    axs[3].set_title('B1 Field Inhomogeneity')
    plt.colorbar(im, ax=axs[3])

    sens_maps = phantom.coil_sens.detach().cpu().numpy()
    for i in range(num_sens_to_show):
        sens_mag = np.abs(sens_maps[i, :, :, 0])
        im = axs[4 + i].imshow(sens_mag.T)
        axs[4 + i].set_title(f'Coil Sensitivity {i+1}')
        plt.colorbar(im, ax=axs[4 + i])

    for ax in axs[4 + num_sens_to_show:]:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def display_shots(shots, x_freq_per_shot, y_freq_per_shot, save_path=None):
    R = len(shots)
    fig, axes = plt.subplots(3, R + 1, figsize=(3 * (R + 1), 9))

    for i, shot in enumerate(shots):
        kspace_sos = torch.sqrt(torch.sum(torch.abs(shot) ** 2, dim=-1))
        axes[0, i].imshow(np.log(np.abs(kspace_sos) + 1), cmap='gray')
        axes[0, i].set_title(f'Shot {i + 1} K-space (SoS)')
        axes[0, i].axis('off')

        images_per_coil = []
        for coil in range(shot.shape[-1]):
            spectrum = np.fft.fftshift(shot[:, :, coil])
            space = np.fft.fft2(spectrum)
            space = np.fft.ifftshift(space)
            images_per_coil.append(np.abs(space))
        images_per_coil = np.stack(images_per_coil, axis=-1)
        img_sos = np.sqrt(np.sum(images_per_coil ** 2, axis=-1))

        axes[1, i].imshow(img_sos, cmap='gray')
        axes[1, i].set_title(f'Shot {i + 1} Image (SoS)')
        axes[1, i].axis('off')

        x_traj = x_freq_per_shot[i]
        y_traj = y_freq_per_shot[i]
        mask = np.abs(shot[:, :, 0]) > 0
        if torch.any(mask):
            axes[2, i].scatter(x_traj[mask], y_traj[mask], c=range(torch.sum(mask)), cmap='viridis', s=1, alpha=0.7)
        axes[2, i].set_title(f'Shot {i + 1} Trajectory')
        axes[2, i].set_aspect('equal')
        axes[2, i].grid(True, alpha=0.3)

    combined_kspace_per_coil = torch.sum(torch.stack(shots), dim=0)
    combined_kspace_sos = torch.sqrt(torch.sum(torch.abs(combined_kspace_per_coil) ** 2, dim=-1))

    combined_images_per_coil = []
    for coil in range(combined_kspace_per_coil.shape[-1]):
        spectrum = np.fft.fftshift(combined_kspace_per_coil[:, :, coil])
        space = np.fft.fft2(spectrum)
        space = np.fft.ifftshift(space)
        combined_images_per_coil.append(np.abs(space))
    combined_images_per_coil = np.stack(combined_images_per_coil, axis=-1)
    combined_img_sos = np.sqrt(np.sum(combined_images_per_coil ** 2, axis=-1))

    axes[0, R].imshow(np.log(np.abs(combined_kspace_sos) + 1), cmap='gray')
    axes[0, R].set_title('Combined K-space (SoS)')
    axes[0, R].axis('off')

    axes[1, R].imshow(combined_img_sos, cmap='gray')
    axes[1, R].set_title('Combined Image (SoS)')
    axes[1, R].axis('off')

    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
    for i in range(R):
        x_traj = x_freq_per_shot[i]
        y_traj = y_freq_per_shot[i]
        mask = np.abs(shots[i][:, :, 0]) > 0
        if torch.any(mask):
            axes[2, R].scatter(x_traj[mask], y_traj[mask], c=colors[i % len(colors)], s=1, alpha=0.7, label=f'Shot {i+1}')
    axes[2, R].set_title('All Trajectories')
    axes[2, R].set_aspect('equal')
    axes[2, R].grid(True, alpha=0.3)
    axes[2, R].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def display_coil_images(shots, k_coils=4, save_path=None):
    combined = torch.sum(torch.stack(shots), dim=0)
    coil_indices = np.linspace(0, combined.shape[-1] - 1, k_coils, dtype=int)

    fig, axes = plt.subplots(1, k_coils, figsize=(3 * k_coils, 3))
    if k_coils == 1:
        axes = [axes]

    for i, coil_idx in enumerate(coil_indices):
        img = np.abs(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(combined[:, :, coil_idx]))))
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Coil {coil_idx}')
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def display_time_series_shots(time_series_shots, flip_angles, grid_size=(5, 10), save_path=None):
    rows, cols = grid_size
    max_shots = rows * cols
    num_shots = min(len(time_series_shots), max_shots)

    # Convert to numpy and get magnitude/phase
    shots_np = [shot.detach().cpu().numpy() for shot in time_series_shots[:num_shots]]
    mags = [np.abs(shot) for shot in shots_np]
    phases = [np.angle(shot) for shot in shots_np]

    # Get global min/max for consistent scaling
    mag_min, mag_max = min(mag.min() for mag in mags), max(mag.max() for mag in mags)
    phase_min, phase_max = min(phase.min() for phase in phases), max(phase.max() for phase in phases)

    # Create 2 rows of subplots: magnitude on top, phase on bottom
    fig, axes = plt.subplots(2 * rows, cols, figsize=(2 * cols, 4 * rows))

    for i in range(num_shots):
        row, col = i // cols, i % cols

        # Magnitude (top half)
        mag_img = mags[i]
        q1_mag = np.quantile(mag_img, 0.01)
        q99_mag = np.quantile(mag_img, 0.99)

        axes[row, col].imshow(mag_img.T, cmap='gray', vmin=mag_min, vmax=mag_max)
        axes[row, col].set_title(f'MAG TS{i + 1} FA={flip_angles[i]}°\n1%:{q1_mag:.1f} 99%:{q99_mag:.1f}', fontsize=9)
        axes[row, col].axis('off')

        # Phase (bottom half)
        phase_img = phases[i]
        q1_phase = np.quantile(phase_img, 0.01)
        q99_phase = np.quantile(phase_img, 0.99)

        axes[row + rows, col].imshow(phase_img.T, cmap='hsv', vmin=phase_min, vmax=phase_max)
        axes[row + rows, col].set_title(f'PHASE TS{i + 1} FA={flip_angles[i]}°\n1%:{q1_phase:.2f} 99%:{q99_phase:.2f}',
                                        fontsize=9)
        axes[row + rows, col].axis('off')

    # Turn off unused subplots
    for i in range(num_shots, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
        axes[row + rows, col].axis('off')

    plt.suptitle(f'MRF Time Series: {num_shots} time points (Magnitude & Phase)', fontsize=16)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()



def plot_calibration_image_vs_first_time_step(calibration_data, time_series_shots, plots_output_path):
    # Convert to numpy and get magnitude/phase
    calib_img = calibration_data.squeeze().detach().cpu().numpy()
    time_img = time_series_shots[0].detach().cpu().squeeze().numpy()

    # Magnitude images
    calib_mag = np.abs(calib_img)
    time_mag = np.abs(time_img)
    mag_diff = calib_mag - time_mag

    # Phase images
    calib_phase = np.angle(calib_img)
    time_phase = np.angle(time_img)
    phase_diff = calib_phase - time_phase

    # Min/max for consistent scaling
    mag_min = min(calib_mag.min(), time_mag.min())
    mag_max = max(calib_mag.max(), time_mag.max())
    phase_min = min(calib_phase.min(), time_phase.min())
    phase_max = max(calib_phase.max(), time_phase.max())

    # Create subplot with 6 images (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Top row: Magnitude
    axes[0, 0].imshow(calib_mag.T, cmap='gray', vmin=mag_min, vmax=mag_max)
    axes[0, 0].set_title('Calibration Magnitude')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(time_mag.T, cmap='gray', vmin=mag_min, vmax=mag_max)
    axes[0, 1].set_title('Time Series Magnitude [0]')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(mag_diff.T, cmap='RdBu_r')
    axes[0, 2].set_title('Magnitude Difference')
    axes[0, 2].axis('off')

    # Bottom row: Phase
    axes[1, 0].imshow(calib_phase.T, cmap='hsv', vmin=phase_min, vmax=phase_max)
    axes[1, 0].set_title('Calibration Phase')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(time_phase.T, cmap='hsv', vmin=phase_min, vmax=phase_max)
    axes[1, 1].set_title('Time Series Phase [0]')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(phase_diff.T, cmap='RdBu_r')
    axes[1, 2].set_title('Phase Difference')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_path, 'calibration_vs_time_complex.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_results(iteration, epochs, losses, T1_gt, T2_gt, PD_gt, B1_gt,
                          t1_pred, t2_pred, pd_pred, b1_pred, real_batch, sim_batch, plots_path,
                          t1_losses, t2_losses, pd_losses, b1_losses):
    # Current loss
    current_loss = losses[-1] if losses else 0

    # Main plot - expanded to 4x5 grid to accommodate B1
    fig, axes = plt.subplots(4, 5, figsize=(25, 16))
    fig.suptitle(f'Iteration {iteration + 1}/{epochs} | Loss: {current_loss:.6f}', fontsize=16)

    # Maps (rows 0-1) - now including B1
    maps = [(T1_gt, t1_pred, 'T1'), (T2_gt, t2_pred, 'T2'), (PD_gt, pd_pred, 'PD'), (B1_gt, b1_pred, 'B1')]

    for i, (gt, pred, name) in enumerate(maps):
        gt_np, pred_np = gt.cpu().numpy(), pred.detach().cpu().numpy()
        vmin, vmax = gt_np.min(), gt_np.max()

        # GT and Prediction
        for j, (img, title) in enumerate([(gt_np, f'GT {name}'), (pred_np, f'Pred {name}')]):
            im = axes[j, i].imshow(img.T, cmap='viridis', vmin=vmin, vmax=vmax)
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
    vmin = real_imgs.min()
    vmax = real_imgs.max()

    for t in range(4):
        for j, (imgs, prefix) in enumerate([(real_imgs, 'Real'), (sim_imgs, 'Sim')]):
            im = axes[j + 2, t].imshow(imgs[t * 5].T, cmap='gray', vmin=vmin, vmax=vmax)
            axes[j + 2, t].set_title(f'{prefix} t={t}')
            axes[j + 2, t].axis('off')

        # Create colorbar on the right side for each time point
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axes[3, t])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    # Hide unused subplots
    axes[0, 4].axis('off')
    axes[1, 4].axis('off')
    axes[2, 4].axis('off')
    axes[3, 4].axis('off')

    plt.tight_layout()
    os.makedirs(f"{plots_path}/training_results", exist_ok=True)
    plt.savefig(f"{plots_path}/training_results/iter_{iteration:04d}.png", dpi=200, facecolor='white')
    plt.close()

    # Loss plots - expanded to 2x3 grid to include B1 losses
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Total loss
    axes[0, 0].semilogy(losses)
    axes[0, 0].set_title(f'Total Loss | Current: {current_loss:.6f}')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss (log scale)')
    axes[0, 0].grid(True, alpha=0.3)

    # Individual component losses - now including B1
    components = [(t1_losses, 'T1', 'red'), (t2_losses, 'T2', 'blue'), (pd_losses, 'PD', 'green'),
                  (b1_losses, 'B1', 'orange')]
    positions = [(0, 1), (0, 2), (1, 0), (1, 1)]

    for (losses_comp, name, color), (row, col) in zip(components, positions):
        current_comp_loss = losses_comp[-1] if losses_comp else 0
        axes[row, col].semilogy(losses_comp, color=color)
        axes[row, col].set_title(f'{name} Loss | Current: {current_comp_loss:.6f}')
        axes[row, col].set_xlabel('Iteration')
        axes[row, col].set_ylabel('Loss (log scale)')
        axes[row, col].grid(True, alpha=0.3)

    # Hide unused subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.join(plots_path, "loss_curves"), exist_ok=True)
    plt.savefig(f"{plots_path}/loss_curves/loss_curve.png", dpi=150, facecolor='white')
    plt.close()



def create_video_from_training_results(training_results_folder, output_video_path, fps=2):
    """Create video from training result images"""
    # Get and sort image files
    image_files = [f for f in os.listdir(training_results_folder) if f.endswith('.png')]
    image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    if not image_files:
        print("No images found")
        return

    # Read first image to get dimensions
    first_img = cv2.imread(os.path.join(training_results_folder, image_files[0]))
    height, width, _ = first_img.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Add each image as a frame
    for img_file in image_files:
        img_path = os.path.join(training_results_folder, img_file)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"Video saved to: {output_video_path}")


def plot_k_images(time_series, k, save_path, title="Time Series"):
    """
    Plot k images from time series and save

    Args:
        time_series: [1, T, H, W] tensor
        k: number of images to plot
        save_path: path to save the plot
        title: plot title
    """
    fig, axes = plt.subplots(1, k, figsize=(k * 3, 3))

    indices = torch.linspace(0, time_series.shape[1] - 1, k).long()

    for i, idx in enumerate(indices):
        axes[i].imshow(time_series[0, idx], cmap='gray')
        axes[i].set_title(f'Frame {idx}')
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()
