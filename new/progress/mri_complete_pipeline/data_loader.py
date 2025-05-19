import h5py
import numpy as np


def load_mat_data(raw_data_path):
    """
    Load raw data and trajectory from .mat file

    Returns:
        rawdata: Complex raw data array (184, 34, 384)
        ktraj_adc: K-space trajectory during ADC (3, 70656)
        t_adc: Time points during ADC (70656,)
        ktraj: Full k-space trajectory (3, 98188)
        t_ktraj: Time points for full trajectory (98188,)
        t_excitation: Excitation times (3,)
        t_refocusing: Refocusing times (6,)
    """
    with h5py.File(raw_data_path, 'r') as f:
        # Load raw data with transpose
        rawdata_real = np.array(f['rawdata']['real'])
        rawdata_imag = np.array(f['rawdata']['imag'])
        rawdata_temp = rawdata_real + 1j * rawdata_imag
        rawdata = rawdata_temp.transpose(2, 1, 0)  # (184, 34, 384)

        # Load trajectory data with transpose for 2D arrays
        ktraj_adc = np.array(f['ktraj_adc']).T  # (3, 70656)
        t_adc = np.array(f['t_adc']).flatten()  # (70656,)
        ktraj = np.array(f['ktraj']).T  # (3, 98188)
        t_ktraj = np.array(f['t_ktraj']).flatten()  # (98188,)
        t_excitation = np.array(f['t_excitation']).flatten()  # (3,)
        t_refocusing = np.array(f['t_refocusing']).flatten()  # (6,)

    return rawdata, ktraj_adc, t_adc, ktraj, t_ktraj, t_excitation, t_refocusing


def load_mr0_data(raw_data_path, seq_file, phantom_path="numerical_brain_cropped.mat"):
    """
    Load trajectory from .mat file and simulate raw data using MR0

    Args:
        raw_data_path: Path to .mat file (for trajectory)
        seq_file: Path to .seq file for MR0 simulation
        phantom_path: Path to phantom .mat file (default: "numerical_brain_cropped.mat")

    Returns:
        rawdata: Simulated complex raw data array (184, 34, 384)
        ktraj_adc: K-space trajectory during ADC (3, 70656)
        t_adc: Time points during ADC (70656,)
        ktraj: Full k-space trajectory (3, 98188)
        t_ktraj: Time points for full trajectory (98188,)
        t_excitation: Excitation times (3,)
        t_refocusing: Refocusing times (6,)
    """
    import MRzeroCore as mr0
    import torch

    with h5py.File(raw_data_path, 'r') as f:
        # Load trajectory data (same as mat data)
        ktraj_adc = np.array(f['ktraj_adc']).T  # (3, 70656)
        t_adc = np.array(f['t_adc']).flatten()  # (70656,)
        ktraj = np.array(f['ktraj']).T  # (3, 98188)
        t_ktraj = np.array(f['t_ktraj']).flatten()  # (98188,)
        t_excitation = np.array(f['t_excitation']).flatten()  # (3,)
        t_refocusing = np.array(f['t_refocusing']).flatten()  # (6,)

        # Get original data dimensions for compatibility
        rawdata_real = np.array(f['rawdata']['real'])
        nADC, nCoils, nAcq = rawdata_real.shape

    # Load MR0 sequence and phantom
    seq0 = mr0.Sequence.import_file(seq_file)
    obj_p = mr0.VoxelGridPhantom.load_mat(phantom_path)
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
    nAcq = total_samples // nADC

    # Reshape signal to match expected rawdata format (nADC, nCoils, nAcq)
    rawdata = signal_np.reshape(nADC, nAcq, actual_coils).transpose(0, 2, 1)  # (nADC, actual_coils, nAcq)

    return rawdata, ktraj_adc, t_adc, ktraj, t_ktraj, t_excitation, t_refocusing


def load_data(raw_data_path, use_mr0=False, seq_file=None, phantom_path="numerical_brain_cropped.mat"):
    """
    Unified data loading function - agnostic to data source

    Args:
        raw_data_path: Path to .mat file
        use_mr0: If True, simulate data with MR0; if False, load from .mat
        seq_file: Path to .seq file (required if use_mr0=True)
        phantom_path: Path to phantom .mat file (only used if use_mr0=True)

    Returns:
        rawdata: Complex raw data array (184, 34, 384)
        ktraj_adc: K-space trajectory during ADC (3, 70656)
        t_adc: Time points during ADC (70656,)
        ktraj: Full k-space trajectory (3, 98188)
        t_ktraj: Time points for full trajectory (98188,)
        t_excitation: Excitation times (3,)
        t_refocusing: Refocusing times (6,)
    """
    if use_mr0:
        if seq_file is None:
            raise ValueError("seq_file must be provided when use_mr0=True")
        return load_mr0_data(raw_data_path, seq_file, phantom_path)
    else:
        return load_mat_data(raw_data_path)