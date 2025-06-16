import torch
import matplotlib
import torch
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import MRzeroCore as mr0
import torch
import torch.fft as fft


def reconstruct_shifted_kspace(kspace_data):
    """
    Reconstruct image from frequency-shifted k-space
    where frequencies range from 0 to 2*kmax
    """
    # Your k-space has DC at center, so:
    # 1. Move DC from center to corner for FFT
    kspace_for_fft = fft.ifftshift(kspace_data, dim=(-2, -1))

    # 2. Apply inverse 2D FFT
    image_complex = fft.ifft2(kspace_for_fft, dim=(-2, -1))

    # 3. Center the resulting image
    image_complex = fft.fftshift(image_complex, dim=(-2, -1))


    return image_complex

def fix_single_shot_epi(kspace_data,ktraj_adc, t_adc):
    """
    Simple odd/even correction for single-shot EPI
    
    kspace_data: torch tensor [phase_lines, freq_encoding]
    """
    print(f"Data shape: {kspace_data.shape} [phase_lines, freq_encoding]")
    
    # Separate odd and even phase encoding lines
    odd_lines = kspace_data[0::2, :]   # Lines 0, 2, 4, ...
    even_lines = kspace_data[1::2, :]  # Lines 1, 3, 5, ...
    
    print(f"Odd lines: {odd_lines.shape}, Even lines: {even_lines.shape}")
    
    # Calculate phase difference using central k-space
    center_line = min(odd_lines.shape[0], even_lines.shape[0]) // 2
    odd_center = odd_lines[center_line, :]
    even_center = even_lines[center_line, :]
    
    correlation = torch.sum(even_center * torch.conj(odd_center))
    phase_diff = torch.angle(correlation)
    
    print(f"Phase difference: {torch.rad2deg(phase_diff):.2f}°")
    
    # Apply correction
    corrected_data = kspace_data.clone()
    corrected_data[1::2, :] *= torch.exp(-1j * phase_diff)
    kspace_data[1::2] = kspace_data[1::2].flip(1)
    corrected_data[1::2] = corrected_data[1::2].flip(1)
    ktraj_adc[:, 1::2, :] = ktraj_adc[:, 1::2, :].flip(-1)

    recon_1 = reconstruct_shifted_kspace(corrected_data)
    recon_1 = torch.abs(recon_1).detach().cpu().numpy()
    recon_2 = reconstruct_shifted_kspace(kspace_data)
    recon_2 = torch.abs(recon_2).detach().cpu().numpy()


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(recon_1, cmap='gray')
    ax1.set_title('Corrected')
    ax2.imshow(recon_2, cmap='gray')
    ax2.set_title('Original')
    plt.show()
