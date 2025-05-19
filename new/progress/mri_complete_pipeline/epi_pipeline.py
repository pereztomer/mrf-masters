import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def calculate_fov_parameters(ktraj_adc, nADC):
    """Calculate FOV and matrix size parameters from k-space trajectory"""
    k_last = ktraj_adc[:, -1]
    k_2last = ktraj_adc[:, -nADC-1]
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
    data_odd = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(rawdata[:,:,0::2], axes=0), axis=0), axes=0)
    data_even = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(rawdata[::-1,:,1::2], axes=0), axis=0), axes=0)

    cmplx_diff = data_even * np.conj(data_odd)
    cmplx_slope = cmplx_diff[1:,:,:] * np.conj(cmplx_diff[:-1,:,:])
    mslope_phs = np.angle(np.sum(cmplx_slope))
    dwell_time = (t_adc[nADC-1] - t_adc[0]) / (nADC - 1)
    measured_traj_delay = mslope_phs / (2 * 2 * np.pi) * nADC * dwell_time
    
    return measured_traj_delay

def resample_data(rawdata, ktraj_adc, t_adc, Nx):
    """Resample raw data to regular Cartesian grid"""
    nADC, nCoils, nAcq = rawdata.shape
    nD = ktraj_adc.shape[0]
    
    # Calculate k-space sampling points
    kxmin = np.min(ktraj_adc[0, :])
    kxmax = np.max(ktraj_adc[0, :])
    kxmax1 = kxmax / (Nx/2 - 1) * (Nx/2)  # compensate for non-symmetric center definition in FFT
    kmaxabs = max(kxmax1, -kxmin)
    kxx = np.arange(-Nx/2, Nx/2) / (Nx/2) * kmaxabs  # kx-sample positions
    
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

def calculate_phase_correction(data_resampled):
    """Calculate phase correction coefficients"""
    data_odd = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(data_resampled[:,:,0::2], axes=0), axis=0), axes=0)
    data_even = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(data_resampled[:,:,1::2], axes=0), axis=0), axes=0)

    cmplx_diff1 = data_even * np.conj(data_odd)
    cmplx_diff2 = data_even[:,:,:-1] * np.conj(data_odd[:,:,1:])

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
            data_pc[i,c,:] = data_resampled[i,c,:] * phase_correction
    
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
    sos_image = np.sqrt(np.sum(np.abs(data_xy)**2, axis=1))
    
    return sos_image, data_xy

def plot_kspace_data(data_resampled):
    """Plot k-space data"""
    plt.figure(figsize=(10, 10))
    plt.imshow(np.abs(data_resampled[:, 0, :]).T, aspect='equal', cmap='gray')
    plt.title('EPI k-space data')
    plt.xlabel('kx samples')
    plt.ylabel('Acquisitions')
    plt.colorbar()
    plt.show()

def plot_phase_data(data_pc):
    """Plot phase of hybrid (x/ky) data"""
    plt.figure(figsize=(10, 8))
    plt.imshow(np.angle(data_pc[:,0,:]).T, aspect='equal', cmap='hsv')
    plt.title('phase of hybrid (x/ky) data')
    plt.xlabel('kx samples')
    plt.ylabel('Acquisitions')
    plt.colorbar()
    plt.show()

def plot_images(sos_image):
    """Plot reconstructed images"""
    num_images = sos_image.shape[2]
    cols = int(np.ceil(np.sqrt(num_images)))  # Automatic layout
    rows = int(np.ceil(num_images / cols))

    plt.figure(figsize=(4*cols, 4*rows))
    for i in range(num_images):
        plt.subplot(rows, cols, i+1)
        plt.imshow(sos_image[:,:,i], aspect='equal', cmap='gray')
        plt.axis('off')
        plt.title(f'Image {i+1}')

    plt.suptitle('EPI Reconstruction - All Images', fontsize=16)
    plt.tight_layout()
    plt.show()

def run_epi_pipeline(rawdata, ktraj_adc, t_adc, use_phase_correction=False, show_plots=True):
    """
    Complete EPI reconstruction pipeline
    
    Args:
        rawdata: Complex raw data array
        ktraj_adc: K-space trajectory during ADC
        t_adc: Time points during ADC
        use_phase_correction: Whether to apply phase correction
        show_plots: Whether to display plots
    
    Returns:
        sos_image: Sum-of-squares images
        data_xy: Complex image data for all coils
        measured_traj_delay: Calculated trajectory delay
    """
    nADC = rawdata.shape[0]
    
    # 1. Calculate FOV parameters
    fov, Nx, Ny, Ny_sampled, delta_ky = calculate_fov_parameters(ktraj_adc, nADC)
    
    # 2. Calculate trajectory delay
    measured_traj_delay = calculate_trajectory_delay(rawdata, t_adc, nADC)
    print(f'measured trajectory delay (assuming it is a calibration data set) is {measured_traj_delay:.8e} s')
    print('type this value in the section above and re-run the script')
    
    # 3. Resample data to Cartesian grid
    data_resampled, ktraj_resampled, t_adc_resampled = resample_data(rawdata, ktraj_adc, t_adc, Nx)
    
    if show_plots:
        plot_kspace_data(data_resampled)
    
    # 4. Calculate and apply phase correction
    mphase1, mphase2, mphase = calculate_phase_correction(data_resampled)
    
    pc_coef = 0
    if use_phase_correction:
        pc_coef = mphase1 / (2 * np.pi)
    
    data_pc = apply_phase_correction(data_resampled, pc_coef)
    
    if show_plots:
        plot_phase_data(data_pc)
    
    # 5. Reconstruct images
    sos_image, data_xy = reconstruct_images(data_pc, Ny_sampled, Ny)
    
    if show_plots:
        plot_images(sos_image)
    
    return sos_image, data_xy, measured_traj_delay