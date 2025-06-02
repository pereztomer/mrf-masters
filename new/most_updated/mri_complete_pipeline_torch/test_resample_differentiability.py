import torch
import numpy as np

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