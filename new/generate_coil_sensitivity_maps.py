import numpy as np
import matplotlib
matplotlib.use('TkAgg')


def generate_coil_maps(matrix_size, num_coils, coil_positions=None):
    """
    Generate real-valued coil sensitivity maps between 0 and 1.

    Args:
        matrix_size (tuple): Size of the sensitivity maps (height, width)
        num_coils (int): Number of coil elements
        coil_positions (list): Optional list of (x,y) coil positions

    Returns:
        ndarray: Real sensitivity maps of shape (num_coils, height, width)
    """
    if coil_positions is None:
        angles = np.linspace(0, 2 * np.pi, num_coils, endpoint=False)
        radius = max(matrix_size) / 2.5
        coil_positions = [(radius * np.cos(angle) + matrix_size[1] / 2,
                           radius * np.sin(angle) + matrix_size[0] / 2)
                          for angle in angles]

    x = np.arange(matrix_size[1])
    y = np.arange(matrix_size[0])
    xx, yy = np.meshgrid(x, y)

    maps = np.zeros((num_coils, matrix_size[0], matrix_size[1]))
    decay_factor = 2/3 * matrix_size[0]  # Increased for wider sensitivity profiles

    for i, (coil_x, coil_y) in enumerate(coil_positions):
        distance = np.sqrt((xx - coil_x) ** 2 + (yy - coil_y) ** 2)
        maps[i] = 1 / (1 + (distance / decay_factor) ** 2)

    # Normalize maps between 0 and 1
    maps = maps / np.max(maps)

    # Ensure sum of squares equals 1 at each pixel
    # maps = maps / np.sqrt(np.sum(maps ** 2, axis=0, keepdims=True))

    return maps


def visualize_maps(maps):
    """
    Visualize the sensitivity maps.
    """
    import matplotlib.pyplot as plt

    num_coils = maps.shape[0]
    cols = int(np.ceil(np.sqrt(num_coils)))
    rows = int(np.ceil(num_coils / cols))

    plt.figure(figsize=(12, 8))
    for i in range(num_coils):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(maps[i], cmap='viridis')
        plt.axis('off')
        plt.title(f'Coil {i + 1}')

    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    maps = generate_coil_maps((32, 32), 8)
    visualize_maps(maps)