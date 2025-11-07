import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

np.random.seed(42)
def create_gaussian_sensitivities(matrix_size, num_coils, coil_positions=None, sigma=0.3):
    """
    Create sensitivity maps using Gaussian profiles
    
    Args:
        matrix_size: Size of the image matrix (assumes square)
        num_coils: Number of coil elements
        coil_positions: List of (x, y) positions for coils. If None, arranged in circle
        sigma: Standard deviation of Gaussian (relative to FOV)
    
    Returns:
        sensitivities: Complex sensitivity maps (matrix_size, matrix_size, num_coils)
    """
    # Create coordinate system
    x = np.linspace(-1, 1, matrix_size)
    y = np.linspace(-1, 1, matrix_size)
    X, Y = np.meshgrid(x, y)

    # Initialize sensitivity array
    sensitivities = np.zeros((matrix_size, matrix_size, num_coils), dtype=complex)

    # Default coil positions in a circle if not provided
    if coil_positions is None:
        coil_radius = 0.9
        coil_angles = np.linspace(0, 2 * np.pi, num_coils, endpoint=False)
        coil_positions = [(coil_radius * np.cos(angle), coil_radius * np.sin(angle))
                          for angle in coil_angles]

    for i, (coil_x, coil_y) in enumerate(coil_positions):
        # Gaussian sensitivity profile
        r_squared = (X - coil_x) ** 2 + (Y - coil_y) ** 2
        magnitude = np.exp(-r_squared / (2 * sigma ** 2))

        # Add smooth phase variation
        phase = 0.3 * np.sin(2 * np.pi * (X - coil_x)) * np.cos(2 * np.pi * (Y - coil_y))

        # Create complex sensitivity
        sensitivities[:, :, i] = magnitude * np.exp(1j * phase)

    return sensitivities


def normalize_sensitivities(sensitivities):
    """
    Normalize sensitivity maps using sum-of-squares
    
    Args:
        sensitivities: Complex sensitivity maps (matrix_size, matrix_size, num_coils)
    
    Returns:
        normalized_sensitivities: Normalized complex sensitivity maps
    """
    # Calculate sum of squares magnitude
    sos = np.sqrt(np.sum(np.abs(sensitivities) ** 2, axis=2))

    # Avoid division by zero
    sos = np.where(sos == 0, 1, sos)

    # Normalize each coil
    normalized_sensitivities = sensitivities / sos[:, :, np.newaxis]

    return normalized_sensitivities


def plot_sensitivities(sensitivities, title="Sensitivity Maps", figsize=(15, 10)):
    """
    Plot sensitivity maps (magnitude and phase)
    
    Args:
        sensitivities: Complex sensitivity maps (matrix_size, matrix_size, num_coils)
        title: Title for the plot
        figsize: Figure size tuple
    """
    num_coils = sensitivities.shape[2]
    rows = 2  # One row for magnitude, one for phase
    cols = min(num_coils, 8)  # Limit columns for display

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(min(num_coils, cols)):
        # Plot magnitude
        im1 = axes[0, i].imshow(np.abs(sensitivities[:, :, i]), cmap='gray', aspect='equal')
        axes[0, i].set_title(f'Coil {i + 1} - Magnitude')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046)

        # Plot phase
        im2 = axes[1, i].imshow(np.angle(sensitivities[:, :, i]), cmap='hsv', aspect='equal')
        axes[1, i].set_title(f'Coil {i + 1} - Phase')
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046)

    # Hide unused subplots
    for i in range(min(num_coils, cols), cols):
        axes[0, i].axis('off')
        axes[1, i].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def save_sensitivities(sensitivities, filename):
    """
    Save sensitivity maps to file
    
    Args:
        sensitivities: Complex sensitivity maps
        filename: Output filename (.npy)
    """
    np.save(filename, sensitivities)
    print(f"Sensitivity maps saved to {filename}")


def load_sensitivities(filename):
    """
    Load sensitivity maps from file
    
    Args:
        filename: Input filename (.npy)
    
    Returns:
        sensitivities: Complex sensitivity maps
    """
    sensitivities = np.load(filename)
    print(f"Sensitivity maps loaded from {filename}")
    return sensitivities


# Example usage and main function
def main():
    # Parameters
    matrix_size = 128
    num_coils = 8

    print("Creating Gaussian sensitivity maps...")
    gaussian_sens = create_gaussian_sensitivities(matrix_size, num_coils, sigma=0.6)
    gaussian_sens = normalize_sensitivities(gaussian_sens)

    plot_sensitivities(gaussian_sens, "Gaussian Sensitivity Maps")

    # # Save sensitivity maps
    # save_sensitivities(gaussian_sens, "gaussian_sensitivities.npy")    #
    # Example of loading
    # loaded_sens = load_sensitivities("gaussian_sensitivities.npy")
    # print(f"Loaded sensitivity shape: {loaded_sens.shape}")


if __name__ == "__main__":
    main()
