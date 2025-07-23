import numpy as np
from scipy.interpolate import interp1d
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pypulseq as pp
import os
from partial_fourier_recon import pocs_pf


def calculate_fov_parameters(ktraj_adc, nADC):
    """Calculate FOV and matrix size parameters from k-space trajectory"""
    k_last = ktraj_adc[:, -1]
    k_2last = ktraj_adc[:, -nADC - 1]
    delta_ky = k_last[1] - k_2last[1]
    fov = 1 / abs(delta_ky)
    Ny_post = round(abs(k_last[1] / delta_ky))

    if k_last[1] > 0:
        Ny_pre = round(abs(np.min(ktraj_adc[1, :]) / delta_ky))
    else:
        Ny_pre = round(abs(np.max(ktraj_adc[1, :]) / delta_ky))

    Nx = 2 * max(Ny_post, Ny_pre)
    Ny = Nx
    Ny_sampled = Ny_pre + Ny_post + 1

    return fov, Nx, Ny, Ny_sampled, delta_ky


def calculate_trajectory_delay(rawdata, t_adc, nADC):
    """Calculate trajectory delay from calibration data"""
    # Classical phase correction / trajectory delay calculation
    data_odd = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(rawdata[:, :, 0::2], axes=0), axis=0), axes=0)
    data_even = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(rawdata[::-1, :, 1::2], axes=0), axis=0), axes=0)

    cmplx_diff = data_even * np.conj(data_odd)
    cmplx_slope = cmplx_diff[1:, :, :] * np.conj(cmplx_diff[:-1, :, :])
    mslope_phs = np.angle(np.sum(cmplx_slope))
    dwell_time = (t_adc[nADC - 1] - t_adc[0]) / (nADC - 1)
    measured_traj_delay = mslope_phs / (2 * 2 * np.pi) * nADC * dwell_time

    return measured_traj_delay


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
            f_data = interp1d(ktraj_adc2[0, :, a], rawdata[:, c, a], kind='linear',
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


def create_full_kspace(data_resampled, ktraj_resampled, Ny, delta_ky):
    """
    Create full k-space matrix from resampled data using k-space coordinates

    Args:
        data_resampled: Resampled data (Nx, nCoils, Ny_sampled)
        ktraj_resampled: K-space coordinates (3, Nx, Ny_sampled)
        Ny: Full matrix size in y direction
        delta_ky: K-space spacing in y direction

    Returns:
        data_full_kspace: Full k-space with zeros in non-acquired regions
    """

    Nx, nCoils, Ny_sampled = data_resampled.shape

    ky_max = ktraj_resampled[1, Nx // 2, 0]

    # Initialize full k-space with consistent dimension ordering
    data_full_kspace = np.zeros((Nx, nCoils, Ny), dtype=complex)

    # Fill the acquired data
    for row in range(Ny_sampled):
        # Get ky for this acquisition (use middle of readout since it should be constant)
        ky = ktraj_resampled[1, Nx // 2, row]

        # Calculate the ky index directly
        ky_idx = int(np.round((ky_max - ky) / delta_ky))

        # Place the entire readout at this ky position
        data_full_kspace[:, :, ky_idx] = data_resampled[:, :, row]

    return data_full_kspace


def calculate_phase_correction(data_resampled):
    """Calculate phase correction coefficients"""
    data_odd = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(data_resampled[:, :, 0::2], axes=0), axis=0), axes=0)
    data_even = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(data_resampled[:, :, 1::2], axes=0), axis=0), axes=0)

    cmplx_diff1 = data_even * np.conj(data_odd)
    cmplx_diff2 = data_even[:, :, :-1] * np.conj(data_odd[:, :, 1:])

    mphase1 = np.angle(np.sum(cmplx_diff1))
    mphase2 = np.angle(np.sum(cmplx_diff2))
    mphase = np.angle(np.sum(np.concatenate([cmplx_diff1.flatten(), cmplx_diff2.flatten()])))

    return mphase1, mphase2, mphase


def apply_phase_correction(data_resampled, pc_coef):
    """Apply phase correction to resampled data"""
    nCoils = data_resampled.shape[1]
    data_pc = data_resampled.copy()

    for c in range(nCoils):
        for i in range(data_resampled.shape[0]):
            phase_correction = np.exp(1j * 2 * np.pi * pc_coef * (np.arange(data_pc.shape[2]) % 2))
            data_pc[i, c, :] = data_resampled[i, c, :] * phase_correction

    return data_pc


def reconstruct_images(data_pc, Ny_sampled, Ny):
    """Reconstruct images from k-space data"""
    nCoils = data_pc.shape[1]
    nAcq = data_pc.shape[2]
    n4 = nAcq // Ny_sampled

    # Reshape for multiple slices or repetitions
    data_pc = data_pc.reshape([data_pc.shape[0], nCoils, Ny_sampled, n4], order='F')

    # Transform to hybrid x/ky space
    data_xky = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(data_pc, axes=0), axis=0), axes=0)

    # Zero-pad if needed
    if Ny_sampled != Ny:
        data_xky1 = np.zeros((data_pc.shape[0], nCoils, Ny, n4), dtype=complex)
        data_xky1[:, :, (Ny - Ny_sampled):Ny, :] = data_xky
        data_xky = data_xky1

    # Transform to image space
    data_xy = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(data_xky, axes=2), axis=2), axes=2)

    # Calculate sum of squares across coils
    sos_image = np.sqrt(np.sum(np.abs(data_xy) ** 2, axis=1))

    return sos_image, data_xy


def plot_kspace_data(data_pc, output_path=None):
    """Plot phase and magnitude of hybrid (x/ky) data for all coils"""
    n_coils = data_pc.shape[1]

    # Calculate grid dimensions
    n_cols = 2  # One column for phase, one for magnitude
    n_rows = n_coils

    # Create figure with appropriate size
    plt.figure(figsize=(12, 4 * n_rows))

    for coil_idx in range(n_coils):
        # Plot phase
        plt.subplot(n_rows, n_cols, 2 * coil_idx + 1)
        plt.imshow(np.angle(data_pc[:, coil_idx, :]).T, aspect='equal', cmap='hsv')
        plt.title(f'Phase - Coil {coil_idx + 1}')
        plt.xlabel('kx samples')
        plt.ylabel('Acquisitions')
        plt.colorbar()

        # Plot magnitude
        plt.subplot(n_rows, n_cols, 2 * coil_idx + 2)
        plt.imshow(np.abs(data_pc[:, coil_idx, :]).T, aspect='equal', cmap='gray')
        plt.title(f'Magnitude - Coil {coil_idx + 1}')
        plt.xlabel('kx samples')
        plt.ylabel('Acquisitions')
        plt.colorbar()

    plt.tight_layout()

    # Save figure if output directory is provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


def plot_images(sos_image, data_xy, output_path=None):
    """Plot reconstructed images for all coils and the final sum-of-squares image"""
    n_coils = data_xy.shape[1]
    num_images = data_xy.shape[3]  # Number of slices/repetitions

    # Calculate grid dimensions for coil images
    cols = int(np.ceil(np.sqrt(n_coils)))  # Automatic layout for coils
    rows = int(np.ceil(n_coils / cols))

    # Create figure with appropriate size
    plt.figure(figsize=(4 * cols, 4 * rows * (num_images + 1)))  # Extra space for SOS images

    # Plot individual coil images
    for i in range(num_images):
        plt.suptitle(f'Slice/Repetition {i + 1}', fontsize=16, y=0.95)
        for coil_idx in range(n_coils):
            plt.subplot(rows * (num_images + 1), cols, coil_idx + 1 + i * rows * cols)
            plt.imshow(np.abs(data_xy[:, coil_idx, :, i]), aspect='equal', cmap='gray')
            plt.axis('off')
            plt.title(f'Coil {coil_idx + 1}')

    # Plot sum-of-squares images
    for i in range(num_images):
        plt.subplot(rows * (num_images + 1), cols, (i + 1) * rows * cols)
        plt.imshow(sos_image[:, :, i], aspect='equal', cmap='gray')
        plt.axis('off')
        plt.title('Sum-of-Squares')

    plt.tight_layout()

    # Save figure if output directory is provided
    if output_path is not None:
        plt.savefig(output_path, dpi=1000, bbox_inches='tight')

    plt.show()
    plt.close()

    # Save SOS images separately
    if output_path is not None:
        for i in range(num_images):
            plt.imsave(f"{output_path}_SOS_{i + 1}.png", sos_image[:, :, i], cmap='gray')


def run_epi_pipeline(rawdata, use_phase_correction=False, show_plots=True, seq=None, output_dir=None):
    """
    Complete EPI reconstruction pipeline
    
    Args:
        rawdata: Complex raw data array
        use_phase_correction: Whether to apply phase correction
        show_plots: Whether to display plots
        seq_file: Path to sequence file
        output_dir: Directory to save plots (optional)
    
    Returns:
        sos_image: Sum-of-squares images
        data_xy: Complex image data for all coils
        measured_traj_delay: Calculated trajectory delay
    """
    # extract relevant parameters
    nADC = int(seq.get_definition('FrequencyEncodingSteps'))
    Nx = int(seq.get_definition('Nx'))
    Ny = int(seq.get_definition('Ny'))
    Ny_sampled = int(seq.get_definition('NySampled'))

    fov_x, fov_y, fov_z = seq.get_definition('FOV')
    # fov = fov_x
    delta_ky = 1 / fov_y
    ktraj_adc_initial, _, _, _, t_adc_initial = seq.calculate_kspace()

    # fov, Nx, Ny, Ny_sampled, delta_ky = calculate_fov_parameters(ktraj_adc, nADC)

    # 2. Calculate trajectory delay
    measured_traj_delay = calculate_trajectory_delay(rawdata, t_adc_initial, nADC)
    print(f'measured trajectory delay (assuming it is a calibration data set) is {measured_traj_delay:.8e} s')
    print(f'Updating {measured_traj_delay:.8e} delay into ktraj_adc and t_adc')

    ktraj_adc, _, _, _, t_adc = seq.calculate_kspace(trajectory_delay=measured_traj_delay)

    # 3. Resample data to Cartesian grid
    data_resampled, ktraj_resampled, t_adc_resampled = resample_data(rawdata, ktraj_adc, t_adc, Nx)

    # 4. Calculate and apply phase correction
    if use_phase_correction:
        mphase1, mphase2, mphase = calculate_phase_correction(data_resampled)
        pc_coef = mphase1 / (2 * np.pi)
        data_resampled = apply_phase_correction(data_resampled, pc_coef)

    half_fourier = True if seq.get_definition('PartialFourierFactor') < 1 else False
    if half_fourier:
        data_resampled = np.where(data_resampled == 0, 1e-10, data_resampled)
        data_full_kspace = create_full_kspace(data_resampled, ktraj_resampled, Ny, delta_ky)
        data_full_kspace = np.moveaxis(data_full_kspace, 1, 0)
        data_full_kspace = pocs_pf(data_full_kspace, iter=10)
        data_full_kspace = data_full_kspace.squeeze(-1)
        data_full_kspace = np.moveaxis(data_full_kspace, 0, 1)
    else:
        data_full_kspace = data_resampled

    if show_plots:
        # plot_kspace_data(data_pc, os.path.join(output_dir, "kspace_partial.png"))
        plot_kspace_data(data_full_kspace, os.path.join(output_dir, "kspace_full.png"))

    # 5. Reconstruct images
    sos_image, data_xy = reconstruct_images(data_resampled, Ny_sampled, Ny)
    sos_image_full_matrix, data_xy_full_matrix = reconstruct_images(data_full_kspace, Ny, Ny)

    if show_plots:
        # plot_images(sos_image, data_xy, os.path.join(output_dir, 'reconstructed_images_partial.png'))
        plot_images(sos_image_full_matrix, data_xy_full_matrix,
                    os.path.join(output_dir, 'reconstructed_images_full.png'))

    return sos_image, data_xy, measured_traj_delay
