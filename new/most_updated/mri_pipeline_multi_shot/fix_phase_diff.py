import torch
import matplotlib.pyplot as plt


def analyze_odd_even_phases(kspace_shots, coil_to_use=0):
    """
    Analyze phase differences between odd/even lines in each shot (PyTorch version)

    kspace_shots: list of torch tensors with shape [freq_encoding, coils, phase_lines]
    coil_to_use: which coil to use for analysis
    """
    # Detect device from first tensor
    device = kspace_shots[0].device
    dtype = kspace_shots[0].dtype

    phase_diffs = []

    print("=== ODD/EVEN PHASE ANALYSIS (PyTorch) ===")
    print(f"Device: {device}")
    print(f"Data shape: [freq_encoding, coils, phase_lines]")

    for shot_idx, shot_data in enumerate(kspace_shots):
        freq_enc, n_coils, phase_lines = shot_data.shape
        print(f"Shot {shot_idx}: {shot_data.shape}")

        # Select coil data
        if coil_to_use == 'combine':
            # Root sum of squares across coils
            coil_data = torch.sqrt(torch.sum(torch.abs(shot_data) ** 2, dim=1))
            # Keep as complex - convert magnitude back to complex
            coil_data = coil_data.to(dtype=torch.complex64 if dtype == torch.complex64 else torch.complex128)
        else:
            coil_data = shot_data[:, coil_to_use, :]

        # Separate odd and even phase encoding lines
        odd_lines = coil_data[:, 0::2]  # Lines 0, 2, 4, ... (phase encoding)
        even_lines = coil_data[:, 1::2]  # Lines 1, 3, 5, ... (phase encoding)

        print(f"  Odd lines: {odd_lines.shape}, Even lines: {even_lines.shape}")

        # Calculate phase difference using central k-space
        center_line = 16 # need to calc this automatically

        # Cross-correlate central phase encoding lines
        odd_center = odd_lines[:, center_line]  # [freq_encoding]
        even_center = even_lines[:, center_line]  # [freq_encoding]

        odd_center = odd_center[80:-80]
        even_center = even_center[80:-80]
        correlation = torch.sum(even_center * torch.conj(odd_center))
        phase_diff = torch.angle(correlation)

        phase_diffs.append(phase_diff)

        print(f"  Phase diff = {phase_diff:.4f} rad ({torch.rad2deg(phase_diff):.2f}°)")

    # Convert to tensor for easier operations
    phase_diffs_tensor = torch.stack(phase_diffs)

    # Check consistency
    mean_phase = torch.mean(phase_diffs_tensor)
    std_phase = torch.std(phase_diffs_tensor)

    print(f"\nSummary:")
    print(f"Mean phase diff: {mean_phase:.4f} rad ({torch.rad2deg(mean_phase):.2f}°)")
    print(f"Std dev: {std_phase:.4f} rad ({torch.rad2deg(std_phase):.2f}°)")
    print(f"Consistent across shots: {'Yes' if std_phase < 0.1 else 'No'}")

    return phase_diffs_tensor


def correct_odd_even_phases(kspace_shots, phase_diffs, use_global=True):
    """
    Apply odd/even phase correction to multi-coil data (PyTorch version)

    kspace_shots: list of torch tensors [freq_encoding, coils, phase_lines]
    phase_diffs: torch tensor of phase differences
    """
    device = kspace_shots[0].device
    corrected_shots = []

    if use_global:
        correction_phase = torch.mean(phase_diffs)
        print(f"\nUsing global correction: {torch.rad2deg(correction_phase):.2f}°")

    for shot_idx, shot_data in enumerate(kspace_shots):
        phase_corr = correction_phase if use_global else phase_diffs[shot_idx]

        # Clone original data
        corrected_shot = shot_data.clone()

        # Apply correction to even phase encoding lines (all coils)
        correction_factor = torch.exp(-1j * phase_corr)
        corrected_shot[:, :, 1::2] = corrected_shot[:, :, 1::2] * correction_factor

        corrected_shots.append(corrected_shot)
        print(f"  Shot {shot_idx}: applied {torch.rad2deg(phase_corr):.2f}° correction")

    return corrected_shots


def compare_correction(original_shot, corrected_shot, coil=0):
    """
    Quick before/after comparison for one coil (PyTorch version)
    """
    # Take one coil and transpose to [phase_lines, freq_encoding] for FFT
    orig_coil = original_shot[:, coil, :].T  # Now [phase_lines, freq_encoding]
    corr_coil = corrected_shot[:, coil, :].T

    # Reconstruct images using torch FFT
    orig_img = torch.abs(torch.fft.ifft2(orig_coil))
    corr_img = torch.abs(torch.fft.ifft2(corr_coil))

    # Convert to numpy for plotting
    orig_img_np = orig_img.detach().cpu().numpy()
    corr_img_np = corr_img.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.imshow(orig_img_np, cmap='gray')
    ax1.set_title(f'Original (Coil {coil})')
    ax1.axis('off')

    ax2.imshow(corr_img_np, cmap='gray')
    ax2.set_title(f'Corrected (Coil {coil})')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def fix_odd_even_phases(kspace_shots, coil_for_analysis=0):
    """
    Complete odd/even correction workflow for multi-coil data (PyTorch version)

    kspace_shots: list of torch tensors [freq_encoding, coils, phase_lines]
    coil_for_analysis: which coil to use for phase analysis
    """
    print(f"Using coil {coil_for_analysis} for phase analysis")
    print(f"Processing on device: {kspace_shots[0].device}")

    # Analyze using specified coil
    phase_diffs = analyze_odd_even_phases(kspace_shots, coil_for_analysis)

    # Correct all coils
    corrected_shots = correct_odd_even_phases(kspace_shots, phase_diffs)

    # Compare first shot
    print(f"\nShowing before/after for Shot 0, Coil {coil_for_analysis}:")
    compare_correction(kspace_shots[0], corrected_shots[0], coil_for_analysis)

    return corrected_shots, phase_diffs


def check_device_consistency(kspace_shots):
    """
    Verify all tensors are on the same device
    """
    devices = [shot.device for shot in kspace_shots]
    if len(set(devices)) > 1:
        print(f"Warning: Tensors on different devices: {devices}")
        return False
    else:
        print(f"All tensors on device: {devices[0]}")
        return True