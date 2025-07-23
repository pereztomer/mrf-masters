import h5py
import matplotlib.pyplot as plt
import numpy as np
import MRzeroCore as mr0
import torch
import pypulseq as pp


def load_mr0_data_torch(seq_file, phantom_path="numerical_brain_cropped.mat"):
    seq = pp.Sequence()
    seq.read(seq_file)
    Nx = int(seq.get_definition('Nx'))
    Ny = int(seq.get_definition('Ny'))
    NySampled = int(seq.get_definition('NySampled'))
    freq_encoding_steps = int(seq.get_definition('FrequencyEncodingSteps'))
    R = int(seq.get_definition('AccelerationFactor'))

    # Load MR0 sequence and phantom
    seq0 = mr0.Sequence.import_file(seq_file)
    k_space = seq0.get_kspace()
    obj_p = mr0.VoxelGridPhantom.load_mat(phantom_path)
    obj_p = obj_p.interpolate(int(Nx), int(Ny), 1)
    phantom_data = obj_p.PD.squeeze().cpu().numpy()
    obj_p = obj_p.build()

    # Simulate
    graph = mr0.compute_graph(seq0.cuda(), obj_p.cuda(), 200, 1e-3)
    signal = mr0.execute_graph(graph, seq0.cuda(), obj_p.cuda(), print_progress=True)

    # Extract shots
    reference_signal_per_shot = []
    reference_signal = signal[:NySampled * R * freq_encoding_steps]
    for shot_number in range(R):
        raw_ref_shot = reference_signal[shot_number * (NySampled * freq_encoding_steps):(shot_number + 1) * (
                    NySampled * freq_encoding_steps)]
        reshaped_shot = raw_ref_shot.reshape(NySampled, freq_encoding_steps).cpu().numpy()
        reference_signal_per_shot.append(reshaped_shot)

    def estimate_and_correct_phases(reference_signal_per_shot):
        """Estimate and correct phase errors between shots and odd/even lines"""

        corrected_shots = []

        # Central k-space region for phase estimation
        central_region = slice(reference_signal_per_shot[0].shape[0] // 2 - 2,
                               reference_signal_per_shot[0].shape[0] // 2 + 2)

        for shot_idx, shot_data in enumerate(reference_signal_per_shot):
            corrected_shot = shot_data.copy()

            # Apply EPI alternating readout correction first
            corrected_shot[1::2, :] = corrected_shot[1::2, ::-1]

            # Phase correction between odd/even lines within shot
            if shot_idx == 0:  # Use first shot as reference
                reference_even = np.mean(corrected_shot[0::2, :], axis=0)
                reference_odd = np.mean(corrected_shot[1::2, :], axis=0)
                base_phase_diff = np.angle(reference_odd) - np.angle(reference_even)

            # Correct odd/even phase difference
            phase_correction_odd = np.exp(-1j * base_phase_diff)
            corrected_shot[1::2, :] *= phase_correction_odd

            # Inter-shot phase correction
            if shot_idx > 0:  # Correct relative to first shot
                shot_ref = np.mean(corrected_shot[central_region, :], axis=0)
                first_shot_ref = np.mean(corrected_shots[0][central_region, :], axis=0)

                # Estimate phase difference
                phase_diff = np.angle(shot_ref) - np.angle(first_shot_ref)

                # Apply phase correction across entire shot
                phase_correction = np.exp(-1j * phase_diff)
                corrected_shot *= phase_correction

            corrected_shots.append(corrected_shot)

        return corrected_shots

    def restore_conjugate_symmetry(kspace):
        """Restore conjugate symmetry for half-Fourier data"""
        kspace_full = kspace.copy()
        acquired_lines = NySampled * R  # 108

        for i in range(acquired_lines, kspace.shape[0]):
            mirror_idx = kspace.shape[0] - 1 - i
            if mirror_idx >= 0 and mirror_idx < acquired_lines:
                kspace_full[i, :] = np.conj(kspace_full[mirror_idx, ::-1])

        return kspace_full

    # Apply phase corrections
    corrected_shots = estimate_and_correct_phases(reference_signal_per_shot)

    # Half-Fourier Multi-shot EPI Reconstruction with phase correction
    full_kspace_multi = np.zeros((Ny, freq_encoding_steps), dtype=complex)

    for shot_idx, corrected_shot in enumerate(corrected_shots):
        end_idx = shot_idx + NySampled * R
        full_kspace_multi[shot_idx:end_idx:R, :] = corrected_shot

    # Restore conjugate symmetry for half-Fourier
    full_kspace_multi = restore_conjugate_symmetry(full_kspace_multi)
    multi_shot_image = np.fft.fftshift(np.fft.ifft2(full_kspace_multi))

    # Individual shots with phase correction
    shot_images = []
    for shot_idx, corrected_shot in enumerate(corrected_shots):
        full_kspace_single = np.zeros((Ny, freq_encoding_steps), dtype=complex)

        end_idx = shot_idx + NySampled * R
        full_kspace_single[shot_idx:end_idx:R, :] = corrected_shot

        full_kspace_single = restore_conjugate_symmetry(full_kspace_single)
        shot_image = np.fft.fftshift(np.fft.ifft2(full_kspace_single))
        shot_images.append(shot_image)

    # Display results
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2 + R, figsize=(5 * (2 + R), 5))

    # Phantom
    axes[0].imshow(np.abs(phantom_data), cmap='gray')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')

    # Multi-shot result with phase correction
    axes[1].imshow(np.abs(multi_shot_image), cmap='gray')
    axes[1].set_title('Multi-Shot (Phase Corrected)')
    axes[1].axis('off')

    # Individual shots with phase correction
    for i, shot_img in enumerate(shot_images):
        axes[2 + i].imshow(np.abs(shot_img), cmap='gray')
        axes[2 + i].set_title(f'Shot {i} (Phase Corrected)')
        axes[2 + i].axis('off')

    plt.tight_layout()
    plt.show()

    exit()




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
        reference_signal_per_shot, rawdata = load_mr0_data_torch(seq_file_path, phantom_path)
    else:
        print("Raw data reading not implemented yet")
        # raw_data = load_mat_data_torch(raw_data_path, device)

    return reference_signal_per_shot, rawdata, seq
