# phantom_creator.py

import MRzeroCore as mr0
import torch
import matplotlib.pyplot as plt


def create_phantom(Nread=32, Nphase=32):
    """
    Create and prepare a phantom for MRI simulation.

    Parameters:
    -----------
    Nread : int
        Number of readout steps/frequency encoding samples
    Nphase : int
        Number of phase encoding steps

    Returns:
    --------
    phantom : mr0.VoxelGridPhantom
        The prepared phantom object
    """
    sz = [Nread, Nphase]

    # Load a phantom object from file
    phantom = mr0.VoxelGridPhantom.load_mat(r"C:\Users\perez\PycharmProjects\mrf-masters\new\most_updated\numerical_brain_cropped.mat")
    phantom = phantom.interpolate(sz[0], sz[1], 1)

    # Manipulate properties
    phantom.T2dash[:] = 30e-3
    phantom.D *= 0
    phantom.B0 *= 1  # alter the B0 inhomogeneity

    return phantom


def plot_phantom(phantom):
    """
    Plot the phantom properties (PD, T1, T2, B0).

    Parameters:
    -----------
    phantom : mr0.VoxelGridPhantom
        The phantom object to plot
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot PD (convert tensor to numpy for plotting)
    pd_np = phantom.PD.detach().cpu().numpy() if isinstance(phantom.PD, torch.Tensor) else phantom.PD
    im1 = axs[0, 0].imshow(pd_np)
    axs[0, 0].set_title('Proton Density (PD)')
    plt.colorbar(im1, ax=axs[0, 0])

    # Plot T1
    t1_np = phantom.T1.detach().cpu().numpy() if isinstance(phantom.T1, torch.Tensor) else phantom.T1
    im2 = axs[0, 1].imshow(t1_np)
    axs[0, 1].set_title('T1 Relaxation Time (s)')
    plt.colorbar(im2, ax=axs[0, 1])

    # Plot T2
    t2_np = phantom.T2.detach().cpu().numpy() if isinstance(phantom.T2, torch.Tensor) else phantom.T2
    im3 = axs[1, 0].imshow(t2_np)
    axs[1, 0].set_title('T2 Relaxation Time (s)')
    plt.colorbar(im3, ax=axs[1, 0])

    # Plot B0
    b0_np = phantom.B0.detach().cpu().numpy() if isinstance(phantom.B0, torch.Tensor) else phantom.B0
    im4 = axs[1, 1].imshow(b0_np)
    axs[1, 1].set_title('B0 Field Inhomogeneity')
    plt.colorbar(im4, ax=axs[1, 1])

    plt.tight_layout()
    plt.show()

    return fig


def create_phantom_with_custom_T1(T1_map, Nread=32, Nphase=32):
    """
    Create a phantom with a custom T1 map.

    Parameters:
    -----------
    T1_map : torch.Tensor
        The custom T1 map to apply to the phantom
    Nread : int
        Number of readout steps
    Nphase : int
        Number of phase encoding steps

    Returns:
    --------
    phantom : mr0.VoxelGridPhantom
        The phantom object with the custom T1 map
    """
    sz = [Nread, Nphase]

    # Load a phantom object from file
    phantom = mr0.VoxelGridPhantom.load_mat(r"C:\Users\perez\PycharmProjects\mrf-masters\new\most_updated\numerical_brain_cropped.mat")
    phantom = phantom.interpolate(sz[0], sz[1], 1)

    # Manipulate properties
    phantom.T2dash[:] = 30e-3
    phantom.D *= 0
    phantom.B0 *= 1  # alter the B0 inhomogeneity

    # Set the custom T1 map
    if len(T1_map.shape) == 2:
        T1_map = T1_map.unsqueeze(-1)  # Add dimension if needed

    phantom.T1 = T1_map

    return phantom


def create_phantom_with_custom_params(T1_map, T2_map, PD_map, Nread=32, Nphase=32):
    """
    Create a phantom with custom T1, T2, and proton density maps.

    Parameters:
    -----------
    T1_map : torch.Tensor
        The custom T1 map to apply to the phantom (in seconds)
    T2_map : torch.Tensor
        The custom T2 map to apply to the phantom (in seconds)
    PD_map : torch.Tensor
        The custom proton density map to apply to the phantom (normalized)
    Nread : int
        Number of readout steps
    Nphase : int
        Number of phase encoding steps

    Returns:
    --------
    phantom : mr0.VoxelGridPhantom
        The phantom object with the custom parameter maps
    """
    sz = [Nread, Nphase]

    # Load a phantom object from file
    phantom = mr0.VoxelGridPhantom.load_mat('numerical_brain_cropped.mat')
    phantom = phantom.interpolate(sz[0], sz[1], 1)

    # Manipulate properties
    phantom.T2dash[:] = 30e-3
    phantom.D *= 0
    phantom.B0 *= 1  # alter the B0 inhomogeneity

    # Ensure all maps have the correct dimensions
    if len(T1_map.shape) == 2:
        T1_map = T1_map.unsqueeze(-1)  # Add dimension if needed
    if len(T2_map.shape) == 2:
        T2_map = T2_map.unsqueeze(-1)  # Add dimension if needed
    if len(PD_map.shape) == 2:
        PD_map = PD_map.unsqueeze(-1)  # Add dimension if needed

    # Set the custom parameter maps
    phantom.T1 = T1_map
    phantom.T2 = T2_map
    phantom.PD = PD_map

    return phantom

# Example usage
if __name__ == "__main__":
    # Create a default phantom
    phantom = create_phantom()

    # Plot the phantom
    plot_phantom(phantom)

    # Build the phantom for simulation
    obj_p = phantom.build()

    print("Phantom created successfully.")
    print(f"Dimensions: {phantom.PD.shape}")
    print(f"T1 range: {phantom.T1.min().item():.3f}s to {phantom.T1.max().item():.3f}s")
    print(f"T2 range: {phantom.T2.min().item():.3f}s to {phantom.T2.max().item():.3f}s")