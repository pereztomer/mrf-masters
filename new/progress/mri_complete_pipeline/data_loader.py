import h5py
import numpy as np
import MRzeroCore as mr0
import torch
import pypulseq as pp


def load_mat_data(raw_data_path):
    """
    Load raw data and trajectory from .mat file

    Returns:
        rawdata: Complex raw data array (184, 34, 384)
    """
    with h5py.File(raw_data_path, 'r') as f:
        # Load raw data with transpose
        rawdata_real = np.array(f['rawdata']['real'])
        rawdata_imag = np.array(f['rawdata']['imag'])
        rawdata_temp = rawdata_real + 1j * rawdata_imag
        rawdata = rawdata_temp.transpose(2, 1, 0)  # (184, 34, 384)

    return rawdata


def load_mr0_data(seq_file, phantom_path="numerical_brain_cropped.mat", use_coil_maps=False, num_coils=None):
    seq = pp.Sequence()
    seq.read(seq_file)

    Nx = int(seq.get_definition('Nx'))
    Ny = int(seq.get_definition('Ny'))
    freq_encoding_steps = int(seq.get_definition('FrequencyEncodingSteps'))
    # Load MR0 sequence and phantom
    seq0 = mr0.Sequence.import_file(seq_file)
    obj_p = mr0.VoxelGridPhantom.load_mat(phantom_path)
    obj_p = obj_p.interpolate(int(Nx), int(Ny), 1)

    # Add coil sensitivity maps if requested
    if use_coil_maps and num_coils is not None:
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
        obj_p.coil_sens = torch.Tensor(coil_maps)
        obj_p = obj_p.build()

    # Simulate the sequence
    graph = mr0.compute_graph(seq0.cuda(), obj_p.cuda(), 200, 1e-3)
    signal = mr0.execute_graph(graph, seq0.cuda(), obj_p.cuda(), print_progress=True)

    # Convert signal to rawdata format (you may need to adjust this based on signal structure)
    # Signal comes as 2D array: (total_samples, coils) e.g., (70000, 1) or (70000, nCoils)
    signal_np = signal.cpu().numpy()

    # Signal shape is (total_samples, actual_coils)
    total_samples, actual_coils = signal_np.shape

    # Calculate number of acquisitions based on total samples and nADC
    nAcq = total_samples // freq_encoding_steps

    # Reshape signal to match expected rawdata format (nADC, nCoils, nAcq)
    rawdata = signal_np.reshape(int(freq_encoding_steps), nAcq, actual_coils, order='F').transpose(0, 2,
                                                                                              1)  # (nADC, actual_coils, nAcq)

    return rawdata


def load_data(raw_data_path, use_mr0=False, seq_file=None, phantom_path="numerical_brain_cropped.mat",
              use_coil_maps=False, num_coils=None):
    """
    Unified data loading function - agnostic to data source

    Args:
        raw_data_path: Path to .mat file
        use_mr0: If True, simulate data with MR0; if False, load from .mat
        seq_file: Path to .seq file (required if use_mr0=True)
        phantom_path: Path to phantom .mat file (only used if use_mr0=True)
        use_coil_maps: Whether to add coil sensitivity maps (only used if use_mr0=True)
        num_coils: Number of coils for sensitivity maps (only used if use_mr0=True and use_coil_maps=True)

    Returns:
        rawdata: Complex raw data array (184, 34, 384)
    """
    seq = pp.Sequence()
    seq.read(seq_file)
    # k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace(trajectory_delay=traj_recon_delay)
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
    if use_mr0:
        if seq_file is None:
            raise ValueError("seq_file must be provided when use_mr0=True")
        raw_data = load_mr0_data(seq_file, phantom_path, use_coil_maps, num_coils)
    else:
        raw_data = load_mat_data(raw_data_path)

    return raw_data, k_traj_adc, t_adc
