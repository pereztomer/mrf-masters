import MRzeroCore as mr0
import torch
from sensitivity_maps import create_gaussian_sensitivities, normalize_sensitivities
import numpy as np

from plotting_utils import plot_phantom


def create_phantom(Nread, Nphase, phantom_path, num_coils):
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
    phantom = mr0.VoxelGridPhantom.load_mat(phantom_path)
    phantom = phantom.interpolate(sz[0], sz[1], 1)

    # Manipulate properties
    phantom.T2dash[:] = 30e-3
    phantom.D *= 0
    phantom.B0 *= 1  # alter the B0 inhomogeneity

    # Add coil sensitivity maps if requested

    # Create coil sensitivity maps using our sensitivity_maps module
    resolution = (Nread, Nphase)
    sens_maps_2d = create_gaussian_sensitivities(resolution[0], num_coils)
    sens_maps_2d = normalize_sensitivities(sens_maps_2d)

    # Convert to MR0 format: (num_coils, x, y, z)
    coil_maps = np.zeros((num_coils, resolution[0], resolution[1], 1), dtype=complex)
    for c in range(num_coils):
        coil_maps[c, :, :, 0] = sens_maps_2d[:, :, c]

    # Set the coil maps in the phantom
    coil_maps = torch.tensor(coil_maps, dtype=torch.complex64)
    phantom.coil_sens = coil_maps

    return phantom, coil_maps


def create_phantom_with_custom_parameters(T1_map, T2_map, PD_map, Nread, Nphase, phantom_path, coil_maps=None, num_coils=None):
    """
    Create a phantom with a custom T2 map, T2 map and PD map.
    """

    sz = [Nread, Nphase]

    # Load a phantom object from file
    phantom = mr0.VoxelGridPhantom.load_mat(phantom_path)
    phantom = phantom.interpolate(sz[0], sz[1], 1)

    device = torch.device('cuda')

    for attr in ['T1', 'T2', 'PD', 'T2dash', 'D', 'B0', 'B1', 'coil_sens', 'size']:
        tensor = getattr(phantom, attr, None)
        if isinstance(tensor, torch.Tensor):
            setattr(phantom, attr, tensor.to(device))

    # Manipulate properties
    phantom.T2dash[:] = 30e-3
    phantom.D *= 0
    phantom.B0 *= 1  # alter the B0 inhomogeneity

    phantom.T1 = T1_map.unsqueeze(-1)
    phantom.T2 = T2_map.unsqueeze(-1)
    phantom.PD = PD_map.unsqueeze(-1)

    # Set the coil maps in the phantom
    if coil_maps is None and num_coils is not None:
        # Create coil sensitivity maps using our sensitivity_maps module
        resolution = (Nread, Nphase)
        sens_maps_2d = create_gaussian_sensitivities(resolution[0], num_coils)
        sens_maps_2d = normalize_sensitivities(sens_maps_2d)

        # Convert to MR0 format: (num_coils, x, y, z)
        coil_maps = np.zeros((num_coils, resolution[0], resolution[1], 1), dtype=complex)
        for c in range(num_coils):
            coil_maps[c, :, :, 0] = sens_maps_2d[:, :, c]

        # Set the coil maps in the phantom
        coil_maps = torch.tensor(coil_maps, dtype=torch.complex64)
    elif coil_maps is None and num_coils is None:
        raise AttributeError('No coil maps provided or number of coils not specified')

    phantom.coil_sens = coil_maps

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


def main():
    phantom_path = r"C:\Users\perez\OneDrive - Technion\masters\mri_research\code\python\mrf-masters\new\most_updated\numerical_brain_cropped.mat"
    Nx = 512
    Ny = 512
    num_coils = 34
    # Create a default phantom
    phantom = create_phantom(Nx, Ny, phantom_path, num_coils)

    # Plot the phantom
    plot_phantom(phantom)

    # Build the phantom for simulation
    obj_p = phantom.build()

    print("Phantom created successfully.")
    print(f"Dimensions: {phantom.PD.shape}")
    print(f"T1 range: {phantom.T1.min().item():.3f}s to {phantom.T1.max().item():.3f}s")
    print(f"T2 range: {phantom.T2.min().item():.3f}s to {phantom.T2.max().item():.3f}s")


# Example usage
if __name__ == "__main__":
    main()
