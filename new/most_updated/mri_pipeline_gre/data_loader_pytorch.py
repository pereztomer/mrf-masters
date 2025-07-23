import h5py
import numpy as np
import MRzeroCore as mr0
import torch
import pypulseq as pp
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import eqdist_grappa_cuda


def display_shots(shots, x_freq_per_shot, y_freq_per_shot):
    R = len(shots)
    fig, axes = plt.subplots(3, R + 1, figsize=(3 * (R + 1), 9))

    # Individual shots
    for i, shot in enumerate(shots):
        # shot shape: (Nread, Nread, num_coils)

        # Row 1: K-space (SoS magnitude)
        kspace_sos = torch.sqrt(torch.sum(torch.abs(shot) ** 2, dim=-1))  # SoS across coils
        axes[0, i].imshow(np.log(np.abs(kspace_sos) + 1), cmap='gray')
        axes[0, i].set_title(f'Shot {i + 1} K-space (SoS)')
        axes[0, i].axis('off')

        # Row 2: Image space (SoS)
        # FFT each coil separately then combine
        images_per_coil = []
        for coil in range(shot.shape[-1]):
            spectrum = np.fft.fftshift(shot[:, :, coil])
            space = np.fft.fft2(spectrum)
            space = np.fft.ifftshift(space)
            images_per_coil.append(np.abs(space))

        # Create SoS image
        images_per_coil = np.stack(images_per_coil, axis=-1)  # Shape: (Nread, Nread, num_coils)
        img_sos = np.sqrt(np.sum(images_per_coil ** 2, axis=-1))  # SoS across coils

        axes[1, i].imshow(img_sos, cmap='gray')
        axes[1, i].set_title(f'Shot {i + 1} Image (SoS)')
        axes[1, i].axis('off')

        # Row 3: Trajectory (use first coil for mask)
        x_traj = x_freq_per_shot[i]
        y_traj = y_freq_per_shot[i]
        mask = np.abs(shot[:, :, 0]) > 0  # Use first coil for mask
        if torch.any(mask):
            axes[2, i].scatter(x_traj[mask], y_traj[mask], c=range(torch.sum(mask)),
                               cmap='viridis', s=1, alpha=0.7)
        axes[2, i].set_title(f'Shot {i + 1} Trajectory')
        axes[2, i].set_aspect('equal')
        axes[2, i].grid(True, alpha=0.3)

    # Combined results
    # Stack all shots and combine across shots, then SoS across coils
    combined_kspace_per_coil = torch.sum(torch.stack(shots), dim=0)  # Sum across shots
    combined_kspace_sos = torch.sqrt(torch.sum(torch.abs(combined_kspace_per_coil) ** 2, dim=-1))  # SoS across coils

    # Create combined SoS image
    combined_images_per_coil = []
    for coil in range(combined_kspace_per_coil.shape[-1]):
        spectrum = np.fft.fftshift(combined_kspace_per_coil[:, :, coil])
        space = np.fft.fft2(spectrum)
        space = np.fft.ifftshift(space)
        combined_images_per_coil.append(np.abs(space))

    combined_images_per_coil = np.stack(combined_images_per_coil, axis=-1)
    combined_img_sos = np.sqrt(np.sum(combined_images_per_coil ** 2, axis=-1))

    # Row 1: Combined K-space (SoS)
    axes[0, R].imshow(np.log(np.abs(combined_kspace_sos) + 1), cmap='gray')
    axes[0, R].set_title('Combined K-space (SoS)')
    axes[0, R].axis('off')

    # Row 2: Combined Image (SoS)
    axes[1, R].imshow(combined_img_sos, cmap='gray')
    axes[1, R].set_title('Combined Image (SoS)')
    axes[1, R].axis('off')

    # Row 3: All trajectories overlaid
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
    for i in range(R):
        x_traj = x_freq_per_shot[i]
        y_traj = y_freq_per_shot[i]
        mask = np.abs(shots[i][:, :, 0]) > 0  # Use first coil for mask
        if torch.any(mask):
            axes[2, R].scatter(x_traj[mask], y_traj[mask],
                               c=colors[i % len(colors)], s=1, alpha=0.7,
                               label=f'Shot {i + 1}')
    axes[2, R].set_title('All Trajectories')
    axes[2, R].set_aspect('equal')
    axes[2, R].grid(True, alpha=0.3)
    axes[2, R].legend()

    plt.tight_layout()
    plt.show()


def display_coil_images(shots, k_coils=4):
    """Display magnitude images from k different coils"""
    # Combine shots (fully sampled)
    combined = torch.sum(torch.stack(shots), dim=0)  # Shape: (Nread, Nread, num_coils)

    # Select k coils evenly spaced
    coil_indices = np.linspace(0, combined.shape[-1] - 1, k_coils, dtype=int)

    fig, axes = plt.subplots(1, k_coils, figsize=(3 * k_coils, 3))
    if k_coils == 1:
        axes = [axes]

    for i, coil_idx in enumerate(coil_indices):
        # FFT and magnitude
        img = np.abs(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(combined[:, :, coil_idx]))))
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Coil {coil_idx}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def display_time_series_shots(time_series_shots, time_series_x_freq, time_series_y_freq, flip_angles,
                              grid_size=(5, 10)):
    """Display time series images in grid layout"""
    rows, cols = grid_size
    max_shots = rows * cols
    num_shots = min(len(time_series_shots), max_shots)

    # Calculate global scale for all images
    all_img_sos = []

    for i in range(num_shots):
        shot = time_series_shots[i]

        # Image SoS
        images_per_coil = []
        for coil in range(shot.shape[-1]):
            img_coil = torch.abs(torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(shot[:, :, coil]))))
            images_per_coil.append(img_coil)
        images_per_coil = torch.stack(images_per_coil, axis=-1)
        img_sos = torch.sqrt(torch.sum(images_per_coil ** 2, axis=-1))
        all_img_sos.append(img_sos)

    # Global min/max
    img_min = min(img.min() for img in all_img_sos)
    img_max = max(img.max() for img in all_img_sos)

    # Create simple grid: one image per subplot
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))

    for i in range(num_shots):
        row = i // cols
        col = i % cols

        # Display image
        axes[row, col].imshow(all_img_sos[i].cpu(), cmap='gray', vmin=img_min, vmax=img_max)
        axes[row, col].set_title(f'TS{i + 1} FA={flip_angles[i]}°', fontsize=10)
        axes[row, col].axis('off')

    # Hide unused subplots
    for i in range(num_shots, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')

    plt.suptitle(f'MRF Time Series: {num_shots} time points', fontsize=16)
    plt.tight_layout()
    plt.show()

def load_mat_data_torch(raw_data_path, device='cpu'):
    """
    Load raw data and trajectory from .mat file

    Args:
        raw_data_path: Path to .mat file
        device: PyTorch device ('cpu' or 'cuda')

    Returns:
        rawdata: Complex raw data array (184, 34, 384) as PyTorch tensor on specified device
    """
    with h5py.File(raw_data_path, 'r') as f:
        # Load raw data with transpose (keep original NumPy loading)
        rawdata_real = np.array(f['rawdata']['real'])
        rawdata_imag = np.array(f['rawdata']['imag'])
        rawdata_temp = rawdata_real + 1j * rawdata_imag
        rawdata_np = rawdata_temp.transpose(2, 1, 0)  # (184, 34, 384)

        # Convert to PyTorch complex tensor and move to device
        # Use torch.complex64 for complex float32 or torch.complex128 for complex float64
        rawdata = torch.from_numpy(rawdata_np).to(device=device, dtype=torch.complex64)

    return rawdata


def sort_data_from_simulator(seq, signal, R, Nread, Nphase_in_practice, fourier_factor, time_steps, num_coils,
                             flip_angles):
    kspace_frequencies = torch.Tensor(seq.calculate_kspace()[0])
    shots = []
    x_freq_per_shot = []
    y_freq_per_shot = []
    for index in range(R):
        single_shot = signal[index * Nread * Nphase_in_practice:(index + 1) * Nread * Nphase_in_practice]
        kspace_shot = kspace_frequencies[:, index * Nread * Nphase_in_practice:(index + 1) * Nread * Nphase_in_practice]

        x_freq_shot = kspace_shot[0].unsqueeze(1)
        y_freq_shot = kspace_shot[1].unsqueeze(1)
        x_freq_shot = torch.reshape(x_freq_shot, (Nphase_in_practice, Nread)).clone().T
        y_freq_shot = torch.reshape(y_freq_shot, (Nphase_in_practice, Nread)).clone().T
        x_freq_shot[:, 0::2] = torch.flip(x_freq_shot[:, 0::2], [0])[:, :]
        y_freq_shot[:, 0::2] = torch.flip(y_freq_shot[:, 0::2], [0])[:, :]
        expanded_x_freq_per_shot = np.zeros((Nread, Nread), dtype=complex)
        expanded_x_freq_per_shot[:, index:int(Nread * fourier_factor):R] = x_freq_shot

        expanded_y_freq_per_shot = np.zeros((Nread, Nread), dtype=complex)
        expanded_y_freq_per_shot[:, index:int(Nread * fourier_factor):R] = y_freq_shot
        x_freq_per_shot.append(expanded_x_freq_per_shot)
        y_freq_per_shot.append(expanded_y_freq_per_shot)

        # Initialize tensor with coils as last dimension
        expanded_kspace_per_shot = torch.zeros((Nread, Nread, num_coils), dtype=torch.complex64)
        # For each coil
        for coil in range(num_coils):
            single_shot_coil = single_shot[:, coil]  # Extract one coil
            single_shot_coil = torch.reshape(single_shot_coil, (Nphase_in_practice, Nread)).clone().T
            single_shot_coil[:, 0::2] = torch.flip(single_shot_coil[:, 0::2], [0])[:, :]
            expanded_kspace_per_shot[:, index: int(Nread * fourier_factor):R, coil] = single_shot_coil

        shots.append(expanded_kspace_per_shot)

    time_series_shots = []
    time_series_x_freq_per_shot = []
    time_series_y_freq_per_shot = []
    for step in range(time_steps):
        kspace_shot = kspace_frequencies[:,
                      (R + step) * Nread * Nphase_in_practice: (R + step + 1) * Nread * Nphase_in_practice]

        x_freq_shot = kspace_shot[0].unsqueeze(1)
        y_freq_shot = kspace_shot[1].unsqueeze(1)

        x_freq_shot = torch.reshape(x_freq_shot, (Nphase_in_practice, Nread)).clone().T
        y_freq_shot = torch.reshape(y_freq_shot, (Nphase_in_practice, Nread)).clone().T

        x_freq_shot[:, 0::2] = torch.flip(x_freq_shot[:, 0::2], [0])[:, :]
        y_freq_shot[:, 0::2] = torch.flip(y_freq_shot[:, 0::2], [0])[:, :]

        expanded_x_freq_per_shot = np.zeros((Nread, Nread), dtype=complex)
        expanded_x_freq_per_shot[:, 0:int(Nread * fourier_factor):R] = x_freq_shot

        expanded_y_freq_per_shot = np.zeros((Nread, Nread), dtype=complex)
        expanded_y_freq_per_shot[:, 0:int(Nread * fourier_factor):R] = y_freq_shot

        time_series_x_freq_per_shot.append(expanded_x_freq_per_shot)
        time_series_y_freq_per_shot.append(expanded_y_freq_per_shot)

        expanded_kspace_per_shot = torch.zeros((Nread, Nread, num_coils), dtype=torch.complex64)

        single_shot = signal[(R + step) * Nread * Nphase_in_practice: (R + step + 1) * Nread * Nphase_in_practice]
        for coil in range(num_coils):
            single_shot_coil = single_shot[:, coil]  # Extract one coil
            single_shot_coil = torch.reshape(single_shot_coil, (Nphase_in_practice, Nread)).clone().T
            single_shot_coil[:, 0::2] = torch.flip(single_shot_coil[:, 0::2], [0])[:, :]
            expanded_kspace_per_shot[:, 0: int(Nread * fourier_factor):R, coil] = single_shot_coil

        time_series_shots.append(expanded_kspace_per_shot)

    block_size = (4, 4)
    acc_factors_2d = (1, 3)
    regularization_factor = 0.1
    device = "cuda"
    calibration_data = torch.sum(torch.stack(shots), dim=0)
    grappa_weights_torch = eqdist_grappa_cuda.GRAPPA_calibrate_weights_2d_torch(calibration_data,
                                                                                acc_factors_2d,
                                                                                device,
                                                                                block_size,
                                                                                regularization_factor)

    for time_step in range(time_steps):
        step = time_series_shots[time_step]
        kspace_recon_kykxc, image_coilcombined_sos, unmixing_map_coilWise = eqdist_grappa_cuda.GRAPPA_interpolate_imageSpace_2d_torch(
            step, acc_factors_2d, block_size, grappa_weights_torch, device)
        time_series_shots[time_step] = kspace_recon_kykxc

    display_shots(shots, x_freq_per_shot, y_freq_per_shot)
    display_time_series_shots(time_series_shots, time_series_x_freq_per_shot, time_series_y_freq_per_shot, flip_angles)
    display_coil_images(shots, k_coils=4)
    print("hi")
    exit()


def load_mr0_data_torch(seq_file, phantom_path="numerical_brain_cropped.mat", num_coils=None):
    seq = pp.Sequence()
    seq.read(seq_file)

    Nx = int(seq.get_definition('Nx'))
    Ny = int(seq.get_definition('Ny'))
    NySampled = int(seq.get_definition('NySampled'))
    R = int(seq.get_definition('AccelerationFactor'))
    flip_angles = seq.get_definition('FlipAngles')
    use_multi_shot = bool(seq.get_definition('MultiShotReference'))
    fourier_factor = seq.get_definition("PartialFourierFactor")
    time_steps = int(seq.get_definition("TimeSteps"))
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

    sort_data_from_simulator(seq, signal, R, Nx, NySampled, fourier_factor, time_steps, num_coils, flip_angles)
    # Signal shape is (total_samples, actual_coils)
    total_samples, actual_coils = signal.shape

    # Calculate number of acquisitions based on total samples and nADC
    nAcq = total_samples // freq_encoding_steps

    signal = signal.contiguous()

    # Transpose first to simulate column-major ('F') flattening
    signal_t = signal.T  # Now shape: (num_coils, num_samples)

    # Then reshape and permute
    rawdata = signal_t.reshape(actual_coils, nAcq, freq_encoding_steps).permute(2, 0, 1)

    return rawdata


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
        raw_data = load_mr0_data_torch(seq_file_path, phantom_path, num_coils)
    else:
        raw_data = load_mat_data_torch(raw_data_path, device)

    return raw_data, seq
