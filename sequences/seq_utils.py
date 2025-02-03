import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_signal(signal, lines_acquired, Nx, R):
    # MR IMAGE RECONSTRUCTION
    fig = plt.figure(figsize=(10, 2))

    # Calculate actual acquired lines

    # Reshape signal considering partial k-space and acceleration
    kspace = torch.reshape((signal), (lines_acquired, Nx)).clone()

    kspace = insert_zero_rows(kspace, R)

    # Apply fftshift before FFT
    spectrum = torch.fft.fftshift(kspace)
    space = torch.fft.fft2(spectrum)
    space = torch.fft.ifftshift(space)

    # convert to numpy
    # space = space.numpy()
    # Plotting
    plt.subplot(142)
    plt.title('log. k-space')
    plt.imshow(np.log(np.abs(kspace) + 1), cmap='gray')
    # mr0.util.imshow(np.log(np.abs(kspace.numpy()) + 1))  # Add 1 to avoid log(0)

    plt.subplot(143)
    plt.title('FFT-magnitude')
    plt.imshow(np.abs(space.numpy()), cmap='gray')
    # mr0.util.imshow(np.abs(space.numpy()))
    plt.colorbar()

    plt.subplot(144)
    plt.title('FFT-phase')
    plt.imshow(np.angle(space.numpy()), cmap='gray')
    # mr0.util.imshow(np.angle(space.numpy()), vmin=-np.pi, vmax=np.pi)
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def insert_zero_rows(matrix, R):
    """
    Insert R-1 zero rows between each original row in k-space
    Args:
        matrix: Input tensor of shape (rows, cols)
        R: Acceleration factor
    Returns:
        Tensor with R-1 zero rows inserted between each original row
    """
    rows, cols = matrix.shape
    # Create new tensor with R times the rows minus (R-1)
    # We subtract (R-1) because we don't need extra zeros after the last row
    result = torch.zeros((R * rows - (R - 1), cols), dtype=matrix.dtype, device=matrix.device)

    # Fill every R-th row with original data
    result[::R] = matrix
    return result

# Your matrix shape is (128, 256)
# After running this function, the new shape will be (255, 256)
# Where odd-numbered rows (1, 3, 5, etc.) are all zeros

def plot_signal_with_shifting(signal, lines_acquired, Nx, R):
    fig = plt.figure(figsize=(10, 2))

    # Reshape signal
    kspace = torch.reshape((signal), (lines_acquired, Nx)).clone()
    kspace = insert_zero_rows(kspace, R)

    # Apply FFT without pre-shifting
    space = torch.fft.fft2(kspace)
    # space = torch.fft.fftshift(space)  # Only shift once after FFT

    # Plotting
    plt.subplot(142)
    plt.title('log. k-space')
    plt.imshow(np.log(np.abs(kspace) + 1), cmap='gray')

    plt.subplot(143)
    plt.title('FFT-magnitude')
    plt.imshow(np.abs(space.numpy()), cmap='gray')
    plt.colorbar()

    plt.subplot(144)
    plt.title('FFT-phase')
    plt.imshow(np.angle(space.numpy()), cmap='gray')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
