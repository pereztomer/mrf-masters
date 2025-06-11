import torch
from scipy.interpolate import interp1d, CubicSpline
import numpy as np
import matplotlib.pyplot as plt


def resample_data(rawdata, ktraj_adc, t_adc, Nx):
    """Resample raw data to regular Cartesian grid"""
    nADC, nCoils, nAcq = rawdata.shape
    nD = ktraj_adc.shape[0]

    # Calculate k-space sampling points
    kxmin = np.min(ktraj_adc[0, :])
    kxmax = np.max(ktraj_adc[0, :])
    kxmax1 = kxmax / (Nx / 2 - 1) * (Nx / 2)  # compensate for non-symmetric center definition in FFT
    kmaxabs = max(kxmax1, -kxmin)
    kxx = np.arange(-Nx / 2, Nx / 2) / (Nx / 2) * kmaxabs  # kx-sample positions

    # Reshape trajectory and time data
    ktraj_adc2 = ktraj_adc.reshape(ktraj_adc.shape[0], nADC, ktraj_adc.shape[1] // nADC, order='F')
    t_adc2 = t_adc.reshape(nADC, len(t_adc) // nADC, order='F')

    # Initialize output arrays
    data_resampled = np.zeros((len(kxx), nCoils, nAcq), dtype=complex)
    ktraj_resampled = np.zeros((nD, len(kxx), nAcq))
    t_adc_resampled = np.zeros((len(kxx), nAcq))

    # Main resampling loop
    for a in range(nAcq):
        # Interpolate data for all coils
        for c in range(nCoils):
            f_data = interp1d(ktraj_adc2[0, :, a], rawdata[:, c, a], kind='cubic',
                              bounds_error=False, fill_value=0)
            data_resampled[:, c, a] = f_data(kxx)

        # Set kx trajectory (just copy kxx)
        ktraj_resampled[0, :, a] = kxx

        # Interpolate other k-space dimensions
        for d in range(1, nD):
            f_ktraj = interp1d(ktraj_adc2[0, :, a], ktraj_adc2[d, :, a], kind='linear',
                               bounds_error=False, fill_value=np.nan)
            ktraj_resampled[d, :, a] = f_ktraj(kxx)

        # Interpolate time
        f_time = interp1d(ktraj_adc2[0, :, a], t_adc2[:, a], kind='linear',
                          bounds_error=False, fill_value=np.nan)
        t_adc_resampled[:, a] = f_time(kxx)

    return data_resampled, ktraj_resampled, t_adc_resampled


def torch_linear_interp(x_orig, y_orig, x_new):
    """Differentiable linear interpolation in PyTorch"""
    # Use original points directly (already sorted)
    x_points = x_orig
    y_points = y_orig

    # Find the indices for interpolation
    indices = torch.searchsorted(x_points[1:], x_new, right=False)
    indices = torch.clamp(indices, 0, len(x_points) - 2)

    # Get the surrounding points
    x0 = x_points[indices]
    x1 = x_points[indices + 1]
    y0 = y_points[indices]
    y1 = y_points[indices + 1]

    # Compute interpolation weights
    dx = x1 - x0
    dx = torch.where(dx == 0, torch.ones_like(dx), dx)
    t = (x_new - x0) / dx

    # Linear interpolation: y = y0 + t * (y1 - y0)
    result = y0 + t * (y1 - y0)

    # Handle extrapolation (set to NaN)
    mask = (x_new < x_points[0]) | (x_new > x_points[-1])
    if torch.is_complex(result):
        nan_val = torch.tensor(float('nan'), dtype=x_new.dtype, device=result.device)
        result[mask] = torch.complex(nan_val, nan_val)
    else:
        result[mask] = float('nan')

    return result


def eval_cubic_spline_torch(coeffs, breakpoints, x):
    """Differentiable cubic spline evaluation in PyTorch for complex data"""
    # Find which segment each x belongs to
    indices = torch.searchsorted(breakpoints[1:], x, right=False)
    indices = torch.clamp(indices, 0, coeffs.shape[1] - 1)

    # Get local x values (x - x_i)
    dx = x - breakpoints[indices]

    # Evaluate cubic polynomial: c0*dx^3 + c1*dx^2 + c2*dx + c3
    c0 = coeffs[0, indices]  # Highest order coefficient
    c1 = coeffs[1, indices]
    c2 = coeffs[2, indices]
    c3 = coeffs[3, indices]  # Constant term

    result = c0 * dx * dx * dx + c1 * dx * dx + c2 * dx + c3

    # Handle extrapolation (set to 0 for complex data)
    mask = (x < breakpoints[0]) | (x > breakpoints[-1])
    result[mask] = 0.0 + 0.0j

    return result


def torch_spline_interp(x_orig, y_orig, x_new):
    """
    Differentiable spline interpolation: fit with scipy, evaluate with pytorch
    Complex data only version
    """
    # Fit spline with scipy (on CPU)
    x_np = x_orig.detach().cpu().numpy()
    y_np = y_orig.detach().cpu().numpy()

    # Use CubicSpline directly to get PPoly representation
    # This is equivalent to interp1d with kind='cubic'
    cs = CubicSpline(x_np, y_np, extrapolate=False)

    # Extract PPoly coefficients (always complex)
    coeffs = torch.from_numpy(cs.c).to(x_orig.device, dtype=y_orig.dtype)  # Shape: (4, n_segments)
    breakpoints = torch.from_numpy(cs.x).to(x_orig.device, dtype=torch.float32)  # Breakpoints

    # Evaluate using pytorch (differentiable!)
    return eval_cubic_spline_torch(coeffs, breakpoints, x_new)


def torch_linear_interp_native(x_orig, y_orig, x_new):
    """Fully differentiable linear interpolation in PyTorch"""

    # Find indices for interpolation
    indices = torch.searchsorted(x_orig[1:], x_new, right=False)
    indices = torch.clamp(indices, 0, len(x_orig) - 2)

    # Get surrounding points
    x0 = x_orig[indices]
    x1 = x_orig[indices + 1]
    y0 = y_orig[indices]
    y1 = y_orig[indices + 1]

    # Linear interpolation
    dx = x1 - x0
    dx = torch.where(dx == 0, torch.ones_like(dx), dx)
    t = (x_new - x0) / dx

    result = y0 + t * (y1 - y0)

    # Handle extrapolation by using edge values instead of NaN
    left_mask = x_new < x_orig[0]
    right_mask = x_new > x_orig[-1]

    result[left_mask] = y_orig[0].to(result.dtype)  # Use first value for left extrapolation
    result[right_mask] = y_orig[-1].to(result.dtype)  # Use last value for right extrapolation

    return result


def resample_data_torch_diff(rawdata, ktraj_adc, t_adc, Nx):
    """PyTorch resample using spline interpolation (complete version)"""
    nADC, nCoils, nAcq = rawdata.shape
    nD = ktraj_adc.shape[0]

    # Calculate k-space sampling points
    kxmin = torch.min(ktraj_adc[0, :])
    kxmax = torch.max(ktraj_adc[0, :])
    kxmax1 = kxmax / (Nx / 2 - 1) * (Nx / 2)
    kmaxabs = torch.max(torch.stack([kxmax1, -kxmin]))
    kxx = torch.arange(-Nx // 2, Nx // 2, device=rawdata.device, dtype=torch.float32) / (Nx / 2) * kmaxabs

    # Reshape trajectory
    ktraj_adc_contiguous = ktraj_adc.contiguous()
    ktraj_adc_t = ktraj_adc_contiguous.T
    ktraj_adc2 = ktraj_adc_t.reshape(nAcq, nADC, nD).permute(2, 1, 0)

    # Reshape time data
    t_adc_contiguous = t_adc.contiguous()
    t_adc2 = t_adc_contiguous.reshape(nAcq, nADC).T

    # Initialize output arrays
    data_resampled = torch.empty((len(kxx), nCoils, nAcq), dtype=rawdata.dtype, device=rawdata.device)
    ktraj_resampled = torch.empty((nD, len(kxx), nAcq), dtype=ktraj_adc.dtype, device=rawdata.device)
    t_adc_resampled = torch.empty((len(kxx), nAcq), dtype=t_adc.dtype, device=rawdata.device)

    # Main resampling loop
    for a in range(nAcq):
        kx_orig = ktraj_adc2[0, :, a]

        # Sort by kx position
        sort_indices = torch.argsort(kx_orig)
        kx_sorted = kx_orig[sort_indices]

        # Interpolate data for all coils using cubic spline
        for c in range(nCoils):
            data_orig = rawdata[:, c, a]
            data_sorted = data_orig[sort_indices]
            # data_resampled[:, c, a] = torch_spline_interp(kx_sorted, data_sorted, kxx)
            data_resampled[:, c, a] = torch_linear_interp_native(kx_sorted, data_sorted, kxx)

        # Set kx trajectory (just copy kxx)
        ktraj_resampled[0, :, a] = kxx

        # Interpolate other k-space dimensions using linear interpolation
        for d in range(1, nD):
            ktraj_orig = ktraj_adc2[d, :, a]
            ktraj_sorted = ktraj_orig[sort_indices]
            ktraj_resampled[d, :, a] = torch_linear_interp(kx_sorted, ktraj_sorted, kxx)

        # Interpolate time using linear interpolation
        t_orig = t_adc2[:, a]
        t_sorted = t_orig[sort_indices]
        t_adc_resampled[:, a] = torch_linear_interp(kx_sorted, t_sorted, kxx)

    return data_resampled, ktraj_resampled, t_adc_resampled
