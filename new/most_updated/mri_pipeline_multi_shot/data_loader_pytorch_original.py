import h5py
import numpy as np
import MRzeroCore as mr0
import torch
import pypulseq as pp


def load_mat_data_torch(seq_file_path, raw_data_path, device='cpu'):
    """
    Load raw data and trajectory from .mat file

    Args:
        raw_data_path: Path to .mat file
        device: PyTorch device ('cpu' or 'cuda')

    Returns:
        rawdata: Complex raw data array (184, 34, 384) as PyTorch tensor on specified device
    """
    seq = pp.Sequence()
    seq.read(seq_file_path)

    with h5py.File(raw_data_path, 'r') as f:
        # Load raw data with transpose (keep original NumPy loading)
        rawdata_real = np.array(f['rawdata']['real'])
        rawdata_imag = np.array(f['rawdata']['imag'])
        rawdata_temp = rawdata_real + 1j * rawdata_imag
        rawdata_np = rawdata_temp.transpose(2, 1, 0)  # (184, 34, 384)

        # Convert to PyTorch complex tensor and move to device
        # Use torch.complex64 for complex float32 or torch.complex128 for complex float64
        rawdata = torch.from_numpy(rawdata_np).to(device=device, dtype=torch.complex64)

        shot1 = rawdata[:, :, 0:36]  # Lines 0-35
        shot2 = rawdata[:, :, 36:72]  # Lines 36-71
        shot3 = rawdata[:, :, 72:108]

        reference_signal_per_shot = [shot1, shot2, shot3]

        return reference_signal_per_shot


def load_mr0_data_torch(seq_file, phantom_path="numerical_brain_cropped.mat", num_coils=None):
    seq = pp.Sequence()
    seq.read(seq_file)

    Nx = int(seq.get_definition('Nx'))
    Ny = int(seq.get_definition('Ny'))
    NySampled = int(seq.get_definition('NySampled'))
    freq_encoding_steps = int(seq.get_definition('FrequencyEncodingSteps'))
    R = int(seq.get_definition('AccelerationFactor'))
    use_multi_shot = bool(seq.get_definition('MultiShotReference'))
    # Load MR0 sequence and phantom
    seq0 = mr0.Sequence.import_file(seq_file)
    obj_p = mr0.VoxelGridPhantom.load_mat(phantom_path)
    obj_p = obj_p.interpolate(int(Nx), int(Ny), 1)

    # Add coil sensitivity maps if requested
    if num_coils is not None:
        from sensitivity_maps import create_gaussian_sensitivities, normalize_sensitivities

        # Create coil sensitivity maps using our sensitivity_maps module
        resolution = (int(Nx), int(Ny))
        sens_maps_2d = create_gaussian_sensitivities(resolution[0], num_coils)
        sens_maps_2d = normalize_sensitivities(sens_maps_2d)

        # Convert to MR0 format: (num_coils, x, y, z)
        coil_maps = np.zeros((num_coils, resolution[0], resolution[1], 1), dtype=complex)
        for c in range(num_coils):
            coil_maps[c, :, :, 0] = sens_maps_2d[:, :, c]

        # Set the coil maps in the phantom
        obj_p.coil_sens = torch.tensor(coil_maps, dtype=torch.complex64)

    obj_p = obj_p.build()

    # Simulate the sequence
    graph = mr0.compute_graph(seq0.cuda(), obj_p.cuda(), 200, 1e-3)
    signal = mr0.execute_graph(graph, seq0.cuda(), obj_p.cuda(), print_progress=True)

    reference_signal_per_shot = []
    if use_multi_shot:
        reference_signal = signal[:NySampled * R * freq_encoding_steps]
        for shot_number in range(R):
            raw_ref_shot = reference_signal[shot_number * (NySampled * freq_encoding_steps):(shot_number + 1) * (
                    NySampled * freq_encoding_steps)]
            samples_number, coil_number = raw_ref_shot.shape
            raw_ref_shot = raw_ref_shot.contiguous()

            # Transpose first to simulate column-major ('F') flattening
            raw_ref_shot = raw_ref_shot.T  # Now shape: (num_coils, num_samples)

            # Then reshape and permute
            raw_ref_shot = raw_ref_shot.reshape(coil_number, NySampled, freq_encoding_steps).permute(2, 0, 1)
            reference_signal_per_shot.append(raw_ref_shot)

    return reference_signal_per_shot


def load_data_torch(raw_data_path, use_mr0=False, seq_file_path=None, phantom_path="numerical_brain_cropped.mat",
                    num_coils=None, device='cpu'):
    """
    Unified data loading function - agnostic to data source

    Args:
        raw_data_path: Path to .mat file
        use_mr0: If True, simulate data with MR0; if False, load from .mat
        seq_file_path: Path to .seq file (required if use_mr0=True)
        phantom_path: Path to phantom .mat file (only used if use_mr0=True)
        use_coil_maps: Whether to add coil sensitivity maps (only used if use_mr0=True)
        num_coils: Number of coils for sensitivity maps (only used if use_mr0=True and use_coil_maps=True)
        device: PyTorch device ('cpu' or 'cuda')

    Returns:
        rawdata: Complex raw data array as PyTorch tensor on specified device
        seq: PyPulseq sequence object
    """
    seq = pp.Sequence()
    seq.read(seq_file_path)

    if use_mr0:
        if seq_file_path is None:
            raise ValueError("seq_file must be provided when use_mr0=True")
        reference_signal_per_shot = load_mr0_data_torch(seq_file_path, phantom_path, num_coils)
    else:
        reference_signal_per_shot = load_mat_data_torch(seq_file_path, raw_data_path, device)

    return reference_signal_per_shot, seq
