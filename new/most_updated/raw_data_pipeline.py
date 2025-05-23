import h5py
import numpy as np

raw_data_path = r"C:\Users\perez\Desktop\test\epi\epi_data.mat"

use_phase_correction = False

with h5py.File(raw_data_path, 'r') as f:
    # Load raw data with transpose
    rawdata_real = np.array(f['rawdata']['real'])
    rawdata_imag = np.array(f['rawdata']['imag'])
    rawdata_temp = rawdata_real + 1j * rawdata_imag
    rawdata = rawdata_temp.transpose(2, 1, 0)  # (184, 34, 384)

    # Load trajectory data with transpose for 2D arrays
    ktraj_adc = np.array(f['ktraj_adc']).T  # (3, 70656)
    t_adc = np.array(f['t_adc']).flatten()  # (70656,)
    ktraj = np.array(f['ktraj']).T  # (3, 98188)
    t_ktraj = np.array(f['t_ktraj']).flatten()  # (98188,)
    t_excitation = np.array(f['t_excitation']).flatten()  # (3,)
    t_refocusing = np.array(f['t_refocusing']).flatten()  # (6,)


# Automatic detection of the measurement parameters (FOV, matrix size, etc)
nADC = rawdata.shape[0]
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

################### testing here ######################################

# Classical phase correction / trajectory delay calculation
# here we assume we are dealing with the calibration data

# MATLAB: ifftshift(ifft(ifftshift(...,1)),1)
# The axis=1 in MATLAB corresponds to axis=0 in Python for the first dimension

data_odd = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(rawdata[:,:,0::2], axes=0), axis=0), axes=0)
data_even = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(rawdata[::-1,:,1::2], axes=0), axis=0), axes=0)

cmplx_diff = data_even * np.conj(data_odd)
cmplx_slope = cmplx_diff[1:,:,:] * np.conj(cmplx_diff[:-1,:,:])
mslope_phs = np.angle(np.sum(cmplx_slope))
dwell_time = (t_adc[nADC-1] - t_adc[0]) / (nADC - 1)
measured_traj_delay = mslope_phs / (2 * 2 * np.pi) * nADC * dwell_time

print(f'measured trajectory delay (assuming it is a calibration data set) is {measured_traj_delay:.8e} s')
print('type this value in the section above and re-run the script')

# we do not calculate the constant phase term here because it depends on
# the definitions of the center of k-space and image-space
# analyze the trajectory, resample the data
# here we expect rawdata ktraj_adc loaded (and having the same dimensions)
nCoils = rawdata.shape[1]  # the incoming data order is [kx coils acquisitions]
nAcq = rawdata.shape[2]
nD = ktraj_adc.shape[0]
kxmin = np.min(ktraj_adc[0, :])
kxmax = np.max(ktraj_adc[0, :])
kxmax1 = kxmax / (Nx/2 - 1) * (Nx/2)  # this compensates for the non-symmetric center definition in FFT
kmaxabs = max(kxmax1, -kxmin)
kxx = np.arange(-Nx/2, Nx/2) / (Nx/2) * kmaxabs  # kx-sample positions
ktraj_adc2 = ktraj_adc.reshape(ktraj_adc.shape[0], nADC, ktraj_adc.shape[1] // nADC, order='F')
t_adc2 = t_adc.reshape(nADC, len(t_adc) // nADC, order='F')  # 'F' = Fortran/column-major order


# Resample all data using the corrected reshape and interpolation
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Initialize output arrays
data_resampled = np.zeros((len(kxx), nCoils, nAcq), dtype=complex)
ktraj_resampled = np.zeros((nD, len(kxx), nAcq))
t_adc_resampled = np.zeros((len(kxx), nAcq))

# Ensure we have the correct reshapes
ktraj_adc2 = ktraj_adc.reshape(ktraj_adc.shape[0], nADC, ktraj_adc.shape[1] // nADC, order='F')
t_adc2 = t_adc.reshape(nADC, len(t_adc) // nADC, order='F')

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

# Plot equivalent to MATLAB's imagesc
plt.figure(figsize=(10, 10))
plt.imshow(np.abs(data_resampled[:, 0, :]).T, aspect='equal', cmap='gray')
plt.title('EPI k-space data')
plt.xlabel('kx samples')
plt.ylabel('Acquisitions')
plt.colorbar()
plt.show()


# In some cases (e.g. because of the incorrectly calculated trajectory) phase correction may be needed
# one such case is the use of the frequency shift proportional to gradient
# in combination with the gradient delay and FOV offset in the RO direction
# this calculation is best done with the calibration data, but also seems
# to work with the actual image data
# here we assume we are dealing with the calibration data

data_odd = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(data_resampled[:,:,0::2], axes=0), axis=0), axes=0)
data_even = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(data_resampled[:,:,1::2], axes=0), axis=0), axes=0)

cmplx_diff1 = data_even * np.conj(data_odd)
cmplx_diff2 = data_even[:,:,:-1] * np.conj(data_odd[:,:,1:])

mphase1 = np.angle(np.sum(cmplx_diff1))
mphase2 = np.angle(np.sum(cmplx_diff2))
mphase = np.angle(np.sum(np.concatenate([cmplx_diff1.flatten(), cmplx_diff2.flatten()])))

# Phase correction
pc_coef = 0
if use_phase_correction:
    pc_coef = mphase1 / (2 * np.pi)

data_pc = data_resampled.copy()
for c in range(nCoils):
    for i in range(data_resampled.shape[0]):
        phase_correction = np.exp(1j * 2 * np.pi * pc_coef * (np.arange(data_pc.shape[2]) % 2))
        data_pc[i,c,:] = data_resampled[i,c,:] * phase_correction

# Plot phase of hybrid (x/ky) data
plt.figure(figsize=(10, 8))
plt.imshow(np.angle(data_pc[:,0,:]).T, aspect='equal', cmap='hsv')
plt.title('phase of hybrid (x/ky) data')
plt.xlabel('kx samples')
plt.ylabel('Acquisitions')
plt.colorbar()
plt.show()


# Reshape for multiple slices or repetitions
n4 = nAcq // Ny_sampled
data_pc = data_pc.reshape([data_pc.shape[0], nCoils, Ny_sampled, n4], order='F')

# Display results
data_xky = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(data_pc, axes=0), axis=0), axes=0)

if Ny_sampled != Ny:
    data_xky1 = np.zeros((data_pc.shape[0], nCoils, Ny, n4), dtype=complex)
    data_xky1[:, :, (Ny - Ny_sampled):Ny, :] = data_xky
    data_xky = data_xky1


data_xy = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(data_xky, axes=2), axis=2), axes=2)

# Show image(s)
# Calculate sum of squares across coils
sos_image = np.sqrt(np.sum(np.abs(data_xy)**2, axis=1))

# Display all images on the same plot automatically
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
