import numpy as np

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch

def plot_phantom(phantom, num_sens_to_show=4, save_path=None):
    total_plots = 4 + num_sens_to_show
    ncols = 4
    nrows = (total_plots + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axs = axs.flatten()

    pd_np = phantom.PD.detach().cpu().numpy()
    im = axs[0].imshow(pd_np)
    axs[0].set_title('Proton Density (PD)')
    plt.colorbar(im, ax=axs[0])

    t1_np = phantom.T1.detach().cpu().numpy()
    im = axs[1].imshow(t1_np)
    axs[1].set_title('T1 Relaxation Time (s)')
    plt.colorbar(im, ax=axs[1])

    t2_np = phantom.T2.detach().cpu().numpy()
    im = axs[2].imshow(t2_np)
    axs[2].set_title('T2 Relaxation Time (s)')
    plt.colorbar(im, ax=axs[2])

    b0_np = phantom.B0.detach().cpu().numpy()
    im = axs[3].imshow(b0_np)
    axs[3].set_title('B0 Field Inhomogeneity')
    plt.colorbar(im, ax=axs[3])

    sens_maps = phantom.coil_sens.detach().cpu().numpy()
    for i in range(num_sens_to_show):
        sens_mag = np.abs(sens_maps[i, :, :, 0])
        im = axs[4 + i].imshow(sens_mag)
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
    img_min = min(img.min() for img in time_series_shots)
    img_max = max(img.max() for img in time_series_shots)

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))

    for i in range(num_shots):
        row, col = i // cols, i % cols
        img = time_series_shots[i].detach().cpu()

        # Calculate quantiles
        q1 = torch.quantile(img, 0.01).item()
        q99 = torch.quantile(img, 0.99).item()

        axes[row, col].imshow(img, cmap='gray', vmin=img_min, vmax=img_max)
        axes[row, col].set_title(f'TS{i + 1} FA={flip_angles[i]}°\n1%:{q1:.1f} 99%:{q99:.1f}', fontsize=9)
        axes[row, col].axis('off')

    for i in range(num_shots, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')

    plt.suptitle(f'MRF Time Series: {num_shots} time points', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

