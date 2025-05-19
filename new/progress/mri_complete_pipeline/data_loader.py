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

def load_mr0_data(raw_data_path, phantom=None, phantom_path=None):
    """
    Load trajectory from .mat file and simulate raw data using MR0
    
    Args:
        raw_data_path: Path to .mat file (for trajectory)
        phantom: MR0 phantom object (optional)
        phantom_path: Path to phantom file (optional)
    
    Returns:
        rawdata: Simulated complex raw data array (184, 34, 384)
        ktraj_adc: K-space trajectory during ADC (3, 70656)
        t_adc: Time points during ADC (70656,)
        ktraj: Full k-space trajectory (3, 98188)
        t_ktraj: Time points for full trajectory (98188,)
        t_excitation: Excitation times (3,)
        t_refocusing: Refocusing times (6,)
    """
    try:
        import mr0
    except ImportError:
        raise ImportError("MR0 not installed. Install with: pip install mr0")
    
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
    
    # Create or load phantom
    if phantom is not None:
        phantom_obj = phantom
    elif phantom_path is not None:
        phantom_obj = mr0.load_phantom(phantom_path)
    else:
        # Default simple brain phantom
        phantom_obj = mr0.brain_phantom(matrix_size=128)
    
    # Simulate k-space data using trajectory from .mat file
    rawdata = mr0.simulate_epi(phantom_obj, ktraj_adc.T, t_adc, 
                              num_coils=nCoils, num_acq=nAcq)
    rawdata = rawdata.transpose(2, 1, 0)  # Match original format (184, 34, 384)
    
    return rawdata, ktraj_adc, t_adc, ktraj, t_ktraj, t_excitation, t_refocusing

def load_data(raw_data_path, use_mr0=False, phantom=None, phantom_path=None):
    """
    Unified data loading function - agnostic to data source
    
    Args:
        raw_data_path: Path to .mat file
        use_mr0: If True, simulate data with MR0; if False, load from .mat
        phantom: MR0 phantom object (optional, only used if use_mr0=True)
        phantom_path: Path to phantom file (optional, only used if use_mr0=True)
    
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
        return load_mr0_data(raw_data_path, phantom, phantom_path)
    else:
        return load_mat_data(raw_data_path)