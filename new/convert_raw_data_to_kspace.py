import torch
import numpy as np


def grid_kspace_2d(values, coordinates,crop_frequencies=False,half_fourier=False, grid_size=(256, 256), filling_method='zeros'):
    """
    Grid non-uniform k-space samples onto a uniform 2D grid, handling undersampling.

    Args:
        values (torch.Tensor): Complex k-space values of shape [num_samples]
        coordinates (torch.Tensor): K-space coordinates of shape [num_samples, 4]
                                  where [:, 0:3] are kx, ky, kz coordinates
        grid_size (tuple): Size of the output grid (height, width)
        filling_method (str): Method to handle unsampled points:
                            'zeros' - fill with zeros (default)
                            'nearest' - fill with nearest neighbor
                            'interpolate' - linear interpolation

    Returns:
        torch.Tensor: Gridded k-space data of shape [height, width]
        torch.Tensor: Sampling mask indicating sampled points (1) and gaps (0)
    """
    # Extract kx and ky coordinates (assuming these are the dimensions we want)
    kx = coordinates[:, 0]
    ky = coordinates[:, 1]

    list_kx = list(set(kx.tolist()))
    list_ky = list(set(ky.tolist()))
    # sort list_kx and list_ky
    list_kx.sort()
    list_ky.sort()
    # print(len(list_kx), len(list_ky))
    # # print(f"list_ky: {len(list_ky)}", list_ky)
    # for i in range(len(list_kx)):
    #     print(f"list_kx:{list_kx[i]} list_ky:{list_ky[i]}")
    # exit()
    # extract max and min frequency range
    max_x_frequency = torch.max(torch.abs(kx.min()), torch.abs(kx.max()))
    min_x_frequency = - max_x_frequency

    max_y_frequency = torch.max(torch.abs(ky.min()), torch.abs(ky.max()))
    min_y_frequency = - max_y_frequency

    # if crop_frequencies:
    #     max_freq = min(torch.abs(ky.min()), torch.abs(ky.max()), torch.abs(kx.min()), torch.abs(kx.max()))
    #     max_x_frequency = max_freq
    #     min_x_frequency = -max_freq
    #     max_y_frequency = max_freq
    #     min_y_frequency = -max_freq

        # # drop frequency values that are not in the range (tak their indices first)
        # indices_to_remove = []
        # for i in range(len(kx)):
        #     if kx[i] > max_x_frequency or kx[i] < min_x_frequency or ky[i] > max_y_frequency or ky[i] < min_y_frequency:
        #         indices_to_remove.append(i)
        # # remove indices from both kx and ky
        # kx = torch.tensor([kx[i] for i in range(len(kx)) if i not in indices_to_remove])
        # ky = torch.tensor([ky[i] for i in range(len(ky)) if i not in indices_to_remove])

    # Normalize coordinates to [-1, 1]
    x_indices = ((kx - min_x_frequency) / (max_x_frequency - min_x_frequency) * grid_size[0]).long()
    y_indices = ((ky - min_y_frequency) / (max_y_frequency - min_y_frequency) * grid_size[1]).long()


    max_x_index = torch.max(x_indices)
    min_x_index = torch.min(x_indices)
    max_y_index = torch.max(y_indices)
    min_y_index = torch.min(y_indices)
    # Create empty grid
    grid = torch.zeros(grid_size, dtype=torch.complex64)
    # fill grid with epsilon values between max_x_index...:
    grid.abs().numpy()

    grid[min_y_index:max_y_index+1, min_x_index:max_x_index+1] = 1e-9

    grid_2 = torch.zeros(grid_size, dtype=torch.complex64)
    grid_2[min_y_index:max_y_index + 1, min_x_index:max_x_index + 1] = 1e-9

    grid_3 = torch.zeros(grid_size, dtype=torch.complex64)
    grid_3[min_y_index:max_y_index + 1, min_x_index:max_x_index + 1] = 1e-9

    grid_4 = torch.zeros(grid_size, dtype=torch.complex64)
    grid_4[min_y_index:max_y_index + 1, min_x_index:max_x_index + 1] = 1e-9

    grid_5 = torch.zeros(grid_size, dtype=torch.complex64)
    grid_5[min_y_index:max_y_index + 1, min_x_index:max_x_index + 1] = 1e-9

    # Ensure indices are within bounds
    x_indices = torch.clamp(x_indices, 0, grid_size[1] - 1)
    y_indices = torch.clamp(y_indices, 0, grid_size[0] - 1)

    # Grid the values using scatter_add
    # Create index tensor for scatter operation
    indices = torch.stack([y_indices, x_indices], dim=1)

    for sample_num, (x_coord, y_coord) in enumerate(indices):
        if grid[x_coord, y_coord] == 1e-9:
            grid[x_coord, y_coord] = values[sample_num]
        elif grid_2[x_coord, y_coord] == 1e-9:
            grid_2[x_coord, y_coord] = values[sample_num]
        elif grid_3[x_coord, y_coord] == 1e-9:
            grid_3[x_coord, y_coord] = values[sample_num]
        elif grid_4[x_coord, y_coord] == 1e-9:
            grid_4[x_coord, y_coord] = values[sample_num]
        elif grid_5[x_coord, y_coord] == 1e-9:
            grid_5[x_coord, y_coord] = values[sample_num]
        else:
            print("here")

    # Assuming grid, grid_2, grid_3, and grid_4 are already defined
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a figure with 3 rows and 2 columns (6 subplots total, we'll use 5)
    fig, axs = plt.subplots(3, 2, figsize=(14, 18))

    # Log transform of grid 1 (top-left)
    im1 = axs[0, 0].imshow(np.log(grid.abs().numpy() + 1), cmap='viridis')
    axs[0, 0].set_title('Grid 1 (log scale)')
    plt.colorbar(im1, ax=axs[0, 0], fraction=0.046, pad=0.04)

    # Log transform of grid 2 (top-right)
    im2 = axs[0, 1].imshow(np.log(grid_2.abs().numpy() + 1), cmap='viridis')
    axs[0, 1].set_title('Grid 2 (log scale)')
    plt.colorbar(im2, ax=axs[0, 1], fraction=0.046, pad=0.04)

    # Log transform of grid 3 (middle-left)
    im3 = axs[1, 0].imshow(np.log(grid_3.abs().numpy() + 1), cmap='viridis')
    axs[1, 0].set_title('Grid 3 (log scale)')
    plt.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)

    # Log transform of grid 4 (middle-right)
    im4 = axs[1, 1].imshow(np.log(grid_4.abs().numpy() + 1), cmap='viridis')
    axs[1, 1].set_title('Grid 4 (log scale)')
    plt.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)

    # Log transform of grid 5 (bottom-left)
    im5 = axs[2, 0].imshow(np.log(grid_5.abs().numpy() + 1), cmap='viridis')
    axs[2, 0].set_title('Grid 5 (log scale)')
    plt.colorbar(im5, ax=axs[2, 0], fraction=0.046, pad=0.04)

    # Remove the unused subplot
    fig.delaxes(axs[2, 1])

    # Adjust Grid 5 to be centered in the bottom row
    axs[2, 0].set_position([0.3, axs[2, 0].get_position().y0,
                            axs[2, 0].get_position().width,
                            axs[2, 0].get_position().height])

    # Adjust layout
    plt.tight_layout()
    plt.show()

    return grid


# Example usage:
"""
# Create sample data
num_samples = 1000
values = torch.randn(num_samples, dtype=torch.complex64)
coordinates = torch.randn(num_samples, 4)

# Apply density compensation
compensated_values = apply_density_compensation(values, coordinates)

# Grid the data
gridded_kspace = grid_kspace_2d(compensated_values, coordinates)

# Apply FFT to get the image
image = torch.fft.ifft2(gridded_kspace)
"""
