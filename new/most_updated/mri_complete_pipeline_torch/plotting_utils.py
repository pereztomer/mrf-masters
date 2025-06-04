import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
def plot_images(sos_image, data_xy, output_path=None):
    """Plot reconstructed images for all coils and the final sum-of-squares image"""
    n_coils = data_xy.shape[1]
    num_images = data_xy.shape[3]  # Number of slices/repetitions

    # Calculate grid dimensions for coil images
    cols = int(np.ceil(np.sqrt(n_coils)))  # Automatic layout for coils
    rows = int(np.ceil(n_coils / cols))

    # Create figure with appropriate size
    plt.figure(figsize=(4 * cols, 4 * rows * (num_images + 1)))  # Extra space for SOS images

    # Plot individual coil images
    for i in range(num_images):
        plt.suptitle(f'Slice/Repetition {i + 1}', fontsize=16, y=0.95)
        for coil_idx in range(n_coils):
            plt.subplot(rows * (num_images + 1), cols, coil_idx + 1 + i * rows * cols)
            plt.imshow(np.abs(data_xy[:, coil_idx, :, i]), aspect='equal', cmap='gray')
            plt.axis('off')
            plt.title(f'Coil {coil_idx + 1}')

    # Plot sum-of-squares images
    for i in range(num_images):
        plt.subplot(rows * (num_images + 1), cols, (i + 1) * rows * cols)
        plt.imshow(sos_image[:, :, i], aspect='equal', cmap='gray')
        plt.axis('off')
        plt.title('Sum-of-Squares')

    plt.tight_layout()

    # Save figure if output directory is provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()



def plot_kspace_data(data_pc, output_path=None):
    """Plot phase and magnitude of hybrid (x/ky) data for all coils"""
    n_coils = data_pc.shape[1]

    # Calculate grid dimensions
    n_cols = 2  # One column for phase, one for magnitude
    n_rows = n_coils

    # Create figure with appropriate size
    plt.figure(figsize=(12, 4 * n_rows))

    for coil_idx in range(n_coils):
        # Plot phase
        plt.subplot(n_rows, n_cols, 2 * coil_idx + 1)
        plt.imshow(np.angle(data_pc[:, coil_idx, :]).T, aspect='equal', cmap='hsv')
        plt.title(f'Phase - Coil {coil_idx + 1}')
        plt.xlabel('kx samples')
        plt.ylabel('Acquisitions')
        plt.colorbar()

        # Plot magnitude
        plt.subplot(n_rows, n_cols, 2 * coil_idx + 2)
        plt.imshow(np.abs(data_pc[:, coil_idx, :]).T, aspect='equal', cmap='gray')
        plt.title(f'Magnitude - Coil {coil_idx + 1}')
        plt.xlabel('kx samples')
        plt.ylabel('Acquisitions')
        plt.colorbar()

    plt.tight_layout()

    # Save figure if output directory is provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # plt.show()
    plt.close()