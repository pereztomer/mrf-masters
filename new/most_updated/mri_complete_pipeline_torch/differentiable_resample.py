import torch
import numpy as np

def torch_linear_interp_native(x_orig, y_orig, x_new):
    """Fully differentiable linear interpolation in PyTorch"""
    indices = torch.searchsorted(x_orig[1:], x_new, right=False)
    indices = torch.clamp(indices, 0, len(x_orig) - 2)

    x0, x1 = x_orig[indices], x_orig[indices + 1]
    y0, y1 = y_orig[indices], y_orig[indices + 1]

    dx = torch.where(x1 - x0 == 0, torch.ones_like(x1 - x0), x1 - x0)
    t = (x_new - x0) / dx
    result = y0 + t * (y1 - y0)

    # Handle extrapolation
    mask = (x_new < x_orig[0]) | (x_new > x_orig[-1])
    if torch.is_complex(result):
        nan_val = torch.tensor(float('nan'), dtype=result.real.dtype, device=result.device)
        result[mask] = torch.complex(nan_val, nan_val).to(result.dtype)
    else:
        result[mask] = float('nan')

    return result


def resample_data_torch_differentiable(rawdata, ktraj_adc, t_adc, Nx):
    """Differentiable PyTorch resample using linear interpolation"""
    nADC, nCoils, nAcq = rawdata.shape
    nD = ktraj_adc.shape[0]

    # Calculate k-space sampling points
    kxmin = torch.min(ktraj_adc[0, :])
    kxmax = torch.max(ktraj_adc[0, :])
    kxmax1 = kxmax / (Nx / 2 - 1) * (Nx / 2)
    kmaxabs = torch.max(torch.stack([kxmax1, -kxmin]))
    kxx = torch.arange(-Nx // 2, Nx // 2, device=rawdata.device, dtype=torch.float32) / (Nx / 2) * kmaxabs

    # Reshape trajectory and time
    ktraj_adc2 = ktraj_adc.T.reshape(nAcq, nADC, nD).permute(2, 1, 0)
    t_adc2 = t_adc.reshape(nAcq, nADC).T

    # Initialize output arrays
    data_resampled = torch.empty((len(kxx), nCoils, nAcq), dtype=rawdata.dtype, device=rawdata.device)
    ktraj_resampled = torch.empty((nD, len(kxx), nAcq), dtype=ktraj_adc.dtype, device=rawdata.device)
    t_adc_resampled = torch.empty((len(kxx), nAcq), dtype=t_adc.dtype, device=rawdata.device)

    # Main resampling loop
    for a in range(nAcq):
        kx_orig = ktraj_adc2[0, :, a]
        sort_indices = torch.argsort(kx_orig)
        kx_sorted = kx_orig[sort_indices]

        # Interpolate data for all coils
        for c in range(nCoils):
            data_orig = rawdata[:, c, a]
            data_sorted = data_orig[sort_indices]
            data_resampled[:, c, a] = torch_linear_interp_native(kx_sorted, data_sorted, kxx)

        # Set kx trajectory
        ktraj_resampled[0, :, a] = kxx

        # Interpolate other k-space dimensions
        for d in range(1, nD):
            ktraj_orig = ktraj_adc2[d, :, a]
            ktraj_sorted = ktraj_orig[sort_indices]
            ktraj_resampled[d, :, a] = torch_linear_interp_native(kx_sorted, ktraj_sorted, kxx)

        # Interpolate time
        t_orig = t_adc2[:, a]
        t_sorted = t_orig[sort_indices]
        t_adc_resampled[:, a] = torch_linear_interp_native(kx_sorted, t_sorted, kxx)

    return data_resampled, ktraj_resampled, t_adc_resampled


def create_test_data(device='cpu'):
    """Create simple test data"""
    nADC, nCoils, nAcq, nD = 32, 4, 8, 3
    
    # Create k-space trajectory
    acquisitions = []
    for a in range(nAcq):
        t = torch.linspace(0, 2*np.pi, nADC, device=device)
        angle = 2 * np.pi * a / nAcq
        kx = t * torch.cos(t + angle)
        ky = t * torch.sin(t + angle)
        kz = 0.1 * torch.sin(t + angle)
        acquisitions.append(torch.stack([kx, ky, kz]))
    
    ktraj_adc = torch.cat(acquisitions, dim=1)
    
    # Create time vector
    t_adc = torch.linspace(0, 1e-3, nADC * nAcq, device=device)
    
    # Create complex raw data
    rawdata = torch.randn(nADC, nCoils, nAcq, dtype=torch.complex64, device=device)
    
    return rawdata, ktraj_adc, t_adc


def test_differentiability():
    """Test gradient flow"""
    print("Testing differentiability...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create test data
    rawdata, ktraj_adc, t_adc = create_test_data(device)
    rawdata.requires_grad_(True)
    Nx = 64
    
    print(f"Input shapes: rawdata{rawdata.shape}, ktraj{ktraj_adc.shape}, t{t_adc.shape}")
    
    # Forward pass
    from resample_grid import resample_data_torch_diff
    data_resampled, ktraj_resampled, t_adc_resampled = resample_data_torch_diff(
        rawdata, ktraj_adc, t_adc, Nx
    )
    
    print(f"Output shape: {data_resampled.shape}")
    
    # Test gradients
    loss = torch.sum(torch.abs(data_resampled)**2)
    loss.backward()
    
    grad_exists = rawdata.grad is not None
    grad_norm = torch.norm(rawdata.grad).item() if grad_exists else 0
    has_issues = torch.isnan(rawdata.grad).any() or torch.isinf(rawdata.grad).any() if grad_exists else True
    
    print(f"Loss: {loss.item():.6f}")
    print(f"Gradient exists: {grad_exists}")
    print(f"Gradient norm: {grad_norm:.6f}")
    print(f"Gradient issues: {has_issues}")
    
    if grad_exists and not has_issues and grad_norm > 0:
        print("✅ Gradients OK!")
    else:
        print("❌ Gradient problems!")
    
    return data_resampled


if __name__ == "__main__":
    print("DIFFERENTIABLE RESAMPLING TEST")
    print("=" * 40)
    
    result = test_differentiability()
    
    print("=" * 40)
    print("Test complete!")