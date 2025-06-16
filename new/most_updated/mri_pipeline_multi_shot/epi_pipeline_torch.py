import numpy as np
from scipy.interpolate import interp1d
import pypulseq as pp
import os
from partial_fourier_recon import pocs_pf
import torch
from resample_grid import resample_data_torch_diff
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from plotting_utils import plot_images, plot_kspace_data, plot_single_kspace


def calculate_trajectory_delay_torch(rawdata, t_adc, nADC):
    """Calculate trajectory delay from calibration data - PyTorch version"""

    # Classical phase correction / trajectory delay calculation
    data_odd = torch.fft.ifftshift(torch.fft.ifft(torch.fft.ifftshift(rawdata[:, :, 0::2], dim=0), dim=0), dim=0)
    data_even = torch.fft.ifftshift(
        torch.fft.ifft(torch.fft.ifftshift(torch.flip(rawdata, dims=[0])[:, :, 1::2], dim=0), dim=0), dim=0)

    cmplx_diff = data_even * torch.conj(data_odd)
    cmplx_slope = cmplx_diff[1:, :, :] * torch.conj(cmplx_diff[:-1, :, :])
    mslope_phs = torch.angle(torch.sum(cmplx_slope))
    dwell_time = (t_adc[nADC - 1] - t_adc[0]) / (nADC - 1)
    measured_traj_delay = mslope_phs / (2 * 2 * torch.pi) * nADC * dwell_time
    return measured_traj_delay.item()


def create_full_kspace_torch(data_resampled, ktraj_resampled, Ny, delta_ky):
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
    ky_max = 431.817
    # Initialize full k-space with consistent dimension ordering
    data_full_kspace = torch.empty((Nx, nCoils, Ny), dtype=data_resampled.dtype, device=data_resampled.device)
    data_full_kspace.zero_()

    # Fill the acquired data
    for row in range(Ny_sampled):
        # Get ky for this acquisition (use middle of readout since it should be constant)
        ky = ktraj_resampled[1, Nx // 2, row]

        # Calculate the ky index directly
        ky_idx = int(torch.round((ky_max - ky) / delta_ky))

        # Place the entire readout at this ky position
        data_full_kspace[:, :, ky_idx] = data_resampled[:, :, row]

    return data_full_kspace


def calculate_phase_correction_torch(data_resampled):
    """Calculate phase correction coefficients"""
    # Calculate odd and even data using PyTorch FFT operations

    data_odd = torch.fft.ifftshift(torch.fft.ifft(torch.fft.ifftshift(data_resampled[:, :, 0::2], dim=0), dim=0), dim=0)

    data_even = torch.fft.ifftshift(torch.fft.ifft(torch.fft.ifftshift(data_resampled[:, :, 1::2], dim=0), dim=0),
                                    dim=0)

    # Calculate complex differences
    cmplx_diff1 = data_even * torch.conj(data_odd)
    cmplx_diff2 = data_even[:, :, :-1] * torch.conj(data_odd[:, :, 1:])

    # Calculate phase angles
    mphase1 = torch.angle(torch.sum(cmplx_diff1))
    mphase2 = torch.angle(torch.sum(cmplx_diff2))

    # Concatenate and calculate combined phase
    combined_diff = torch.cat([cmplx_diff1.flatten(), cmplx_diff2.flatten()])
    mphase = torch.angle(torch.sum(combined_diff))

    return mphase1, mphase2, mphase


def apply_phase_correction_torch(data_resampled, pc_coef):
    """Apply phase correction to resampled data - PyTorch version"""
    # Get dimensions
    nSamples, nCoils, nPoints = data_resampled.shape

    # Create the alternating pattern (0, 1, 0, 1, ...)
    indices = torch.arange(nPoints, dtype=torch.float32, device=data_resampled.device)
    alternating_pattern = indices % 2

    # Calculate phase correction: exp(1j * 2 * pi * pc_coef * alternating_pattern)
    phase_arg = 2 * torch.pi * pc_coef * alternating_pattern

    # Create complex exponential: exp(1j * phase_arg) = cos(phase_arg) + 1j * sin(phase_arg)
    phase_correction = torch.complex(torch.cos(phase_arg), torch.sin(phase_arg)).to(data_resampled.dtype)

    # Apply phase correction to all samples and coils at once using broadcasting
    # Shape: [nSamples, nCoils, nPoints] * [nPoints] -> [nSamples, nCoils, nPoints]
    data_pc = data_resampled * phase_correction

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


def reconstruct_images_torch(data_pc, Ny_sampled, Ny):
    """Reconstruct images from k-space data"""
    nCoils = data_pc.shape[1]
    nAcq = data_pc.shape[2]
    n4 = nAcq // Ny_sampled

    # Reshape for multiple slices or repetitions - emulate F-order
    data_pc = data_pc.contiguous()
    torch_flat = data_pc.permute(2, 1, 0).contiguous().flatten()
    data_pc = torch_flat.view(n4, Ny_sampled, nCoils, data_pc.shape[0]).permute(3, 2, 1, 0)

    # Transform to hybrid x/ky space
    data_xky = torch.fft.fftshift(
        torch.fft.ifft(
            torch.fft.ifftshift(data_pc, dim=0),
            dim=0
        ),
        dim=0
    )

    # Zero-pad if needed
    if Ny_sampled != Ny:
        data_xky1 = torch.zeros(
            (data_pc.shape[0], nCoils, Ny, n4),
            dtype=data_xky.dtype,
            device=data_xky.device
        )
        data_xky1[:, :, (Ny - Ny_sampled):Ny, :] = data_xky
        data_xky = data_xky1

    # Transform to image space
    data_xy = torch.fft.fftshift(
        torch.fft.ifft(
            torch.fft.ifftshift(data_xky, dim=2),
            dim=2
        ),
        dim=2
    )

    # Calculate sum of squares across coils
    sos_image = torch.sqrt(torch.sum(torch.abs(data_xy) ** 2, dim=1))

    return sos_image, data_xy


def find_nonzero_rows(kspace_data, threshold=1e-6):
    """Find non-zero rows in 2D k-space data."""
    magnitude = np.abs(kspace_data)
    row_max = np.max(magnitude, axis=0)
    return np.where(row_max > threshold)[0]


def compare_shots(kspace1, kspace2):
    """Compare coverage between two k-space shots."""
    rows1 = find_nonzero_rows(kspace1)
    rows2 = find_nonzero_rows(kspace2)
    overlap = np.intersect1d(rows1, rows2)

    print(f"Shot 1 covers {len(rows1)} rows")
    print(f"Shot 2 covers {len(rows2)} rows")
    print(f"Overlap: {len(overlap)} rows")

    return rows1, rows2, overlap


def image_to_kspace(image):
    """Convert SOS image back to k-space domain."""
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image)))

def run_epi_pipeline_torch(reference_signal_per_shot,
                           rawdata,
                           device,
                           use_phase_correction=False,
                           use_grappa=False,
                           show_plots=True,
                           seq=None,
                           output_dir=None):
    """
    Complete EPI reconstruction pipeline
    
    Args:
        reference_signal_per_shot: Complex reference data (multi shot epi)
        rawdata: Complex raw data array
        use_phase_correction: Whether to apply phase correction
        use_grappa: use grappa for accelerated reconstruction
        show_plots: Whether to display plots
        seq_file: Path to sequence file
        output_dir: Directory to save plots (optional)
    
    Returns:
        sos_image: Sum-of-squares images
        data_xy: Complex image data for all coils
        measured_traj_delay: Calculated trajectory delay
    """
    rawdata.requires_grad_(True)
    # extract relevant parameters
    nADC = int(seq.get_definition('FrequencyEncodingSteps'))
    Nx = int(seq.get_definition('Nx'))
    Ny = int(seq.get_definition('Ny'))
    Ny_sampled = int(seq.get_definition('NySampled'))
    R = int(seq.get_definition('AccelerationFactor'))

    freq_encoding_steps = int(seq.get_definition('FrequencyEncodingSteps'))

    fov_x, fov_y, fov_z = seq.get_definition('FOV')
    # fov = fov_x
    delta_ky = 1 / fov_y
    ktraj_adc_initial, _, _, _, t_adc_initial = seq.calculate_kspace()
    from fix_phase_diff import fix_odd_even_phases
    fix_odd_even_phases(reference_signal_per_shot, coil_for_analysis=0)

    exit()
    reference_kspace = None
    for shot_number in range(len(reference_signal_per_shot)):
        ref_shot = reference_signal_per_shot[shot_number]
        # ktraj_adc_shot = torch.from_numpy(ktraj_adc_initial[:,
        #                                   shot_number * (Ny_sampled * freq_encoding_steps):(shot_number + 1) * (
        #                                               Ny_sampled * freq_encoding_steps)]).to(device)
        # t_adc_initial_shot = torch.from_numpy(t_adc_initial[
        #                                       shot_number * (Ny_sampled * freq_encoding_steps):(shot_number + 1) * (
        #                                                   Ny_sampled * freq_encoding_steps)]).to(device)
        #
        # measured_traj_delay = calculate_trajectory_delay_torch(ref_shot, t_adc_initial_shot, nADC)
        # ktraj_adc, _, _, _, t_adc = seq.calculate_kspace(trajectory_delay=measured_traj_delay)
        # ktraj_adc_shot = torch.from_numpy(ktraj_adc[:,
        #                                   shot_number * (Ny_sampled * freq_encoding_steps):(shot_number + 1) * (
        #                                           Ny_sampled * freq_encoding_steps)]).to(device)
        # t_adc_initial_shot = torch.from_numpy(t_adc[
        #                                       shot_number * (Ny_sampled * freq_encoding_steps):(shot_number + 1) * (
        #                                               Ny_sampled * freq_encoding_steps)]).to(device)
        #
        # ref_shot, ktraj_adc_shot, t_adc_initial_shot = resample_data_torch_diff(ref_shot, ktraj_adc_shot,
        #                                                                         t_adc_initial_shot, Nx)
        # mphase1_torch, mphase2_torch, mphase_torch = calculate_phase_correction_torch(ref_shot)
        # pc_coef_torch = mphase1_torch / (2 * np.pi)
        # ref_shot = apply_phase_correction_torch(ref_shot, pc_coef_torch)
        #
        # ref_shot = create_full_kspace_torch(ref_shot, ktraj_adc_shot, Ny, delta_ky)
        # if reference_kspace is None:
        #     reference_kspace = ref_shot
        # else:
        #     reference_kspace += ref_shot



    sos_image, data_xy = reconstruct_images_torch(reference_kspace, Ny, Ny)
    ref_kspace_single_coil = image_to_kspace(sos_image)
    plot_kspace_data(reference_kspace.detach().cpu().numpy(), os.path.join(output_dir, "kspace_reference.png"))
    plot_images(sos_image.detach().cpu().numpy(), data_xy.detach().cpu().numpy(),
                os.path.join(output_dir, 'reconstructed_images_reference.png'))
    plot_single_kspace(ref_kspace_single_coil,
                       os.path.join(output_dir, 'single_kspace_reference.png'))
    exit()

    # 2. Calculate trajectory delay
    measured_traj_delay = calculate_trajectory_delay_torch(rawdata, t_adc_initial, nADC)
    print(f'measured trajectory delay (assuming it is a calibration data set) is {measured_traj_delay:.8e} s')
    print(f'Updating {measured_traj_delay:.8e} delay into ktraj_adc and t_adc')

    ktraj_adc, _, _, _, t_adc = seq.calculate_kspace(trajectory_delay=measured_traj_delay)

    # Convert to PyTorch tensors
    ktraj_adc = torch.from_numpy(ktraj_adc).to(device)
    t_adc = torch.from_numpy(t_adc).to(device)

    # 3. Resample data to Cartesian grid

    data_resampled, ktraj_resampled, t_adc_resampled = resample_data_torch_diff(rawdata, ktraj_adc, t_adc, Nx)

    # 4. Calculate and apply phase correction
    if use_phase_correction:
        mphase1_torch, mphase2_torch, mphase_torch = calculate_phase_correction_torch(data_resampled)
        pc_coef_torch = mphase1_torch / (2 * np.pi)
        data_resampled = apply_phase_correction_torch(data_resampled, pc_coef_torch)

    half_fourier = True if seq.get_definition('PartialFourierFactor') < 1 else False
    if half_fourier:
        print("Half Fourier not implemented in pyroch yet, only numpy")
        data_resampled = torch.where(data_resampled == 0, 1e-10, data_resampled)
        data_resampled = create_full_kspace_torch(data_resampled, ktraj_resampled, Ny, delta_ky)
        # data_full_kspace = np.moveaxis(data_resampled, 1, 0)
        # data_full_kspace = pocs_pf(data_resampled, iter=10)
        # data_full_kspace = data_resampled.squeeze(-1)
        # data_full_kspace = np.moveaxis(data_resampled, 0, 1)

    if use_grappa:
        print("Yeas")

    sos_image, data_xy = reconstruct_images_torch(data_resampled, Ny, Ny)

    if show_plots:
        plot_kspace_data(data_resampled.detach().cpu().numpy(), os.path.join(output_dir, "kspace_full.png"))
        plot_images(sos_image.detach().cpu().numpy(), data_xy.detach().cpu().numpy(),
                    os.path.join(output_dir, 'reconstructed_images_full.png'))

    return sos_image, data_xy, measured_traj_delay
