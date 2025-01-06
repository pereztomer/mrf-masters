import numpy as np
from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.opts import Opts
import matplotlib.pyplot as plt
import MRzeroCore as mr0
import torch

# System limits and timing
system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130,
              slew_unit='T/m/s', rf_ringdown_time=30e-6,
              rf_dead_time=100e-6, adc_dead_time=20e-6,
              grad_raster_time=10e-6,
              rf_raster_time=1e-6)

# Create a new sequence object
seq = Sequence(system)

# Sequence parameters
fov = 256e-3  # FOV in meters
n = 256  # Matrix size
slice_thickness = 4.5e-3  # Slice thickness in meters
tr_times = np.loadtxt('tr.txt')  # Load TR times in ms and convert to seconds
fa_angles = np.loadtxt('fa.txt')  # Load flip angles
R=2
# Derived parameters
delta_k = 1 / fov
kmax = n / 2 * delta_k
grad_raster = system.grad_raster_time
readout_time = np.ceil(1.024e-3 / grad_raster) * grad_raster
echo_spacing = np.ceil(1.391e-3 / grad_raster) * grad_raster
rf_duration = np.ceil(1e-3 / grad_raster) * grad_raster
te = 18e-3


def make_rf_slice(flip_angle):
    """Create RF and slice select gradients"""
    rf = make_sinc_pulse(flip_angle=flip_angle,
                         system=system,
                         duration=rf_duration,
                         slice_thickness=slice_thickness,
                         apodization=0.5,
                         time_bw_product=4,
                         phase_offset=0,
                         use='excitation')

    gz = make_trapezoid(channel='z',
                        system=system,
                        duration=rf_duration,
                        area=1 / slice_thickness)

    gz_reph_time = np.ceil(1e-3 / grad_raster) * grad_raster
    gz_reph = make_trapezoid(channel='z',
                             system=system,
                             area=-gz.area / 2,
                             duration=gz_reph_time)

    return rf, gz, gz_reph


def make_epi_readout():
    """Create EPI readout gradients"""
    pre_time = np.ceil(1e-3 / grad_raster) * grad_raster
    gx_pre = make_trapezoid(channel='x', system=system,
                            area=-kmax,
                            duration=pre_time)

    gx_flat_time = np.ceil(0.857e-3 / grad_raster) * grad_raster
    gx_ramp_time = np.ceil(0.167e-3 / grad_raster) * grad_raster

    gx_amp = kmax / (readout_time / 2)

    adc = make_adc(num_samples=256,
                   duration=readout_time,
                   delay=gx_ramp_time)

    gx = make_trapezoid(channel='x',
                        system=system,
                        amplitude=gx_amp,
                        flat_time=gx_flat_time,
                        rise_time=gx_ramp_time)

    return gx_pre, gx, adc


def add_epi_block(seq, flip_angle, rf_phase=0):
    """Add a single EPI block to the sequence"""
    # Get block start time (first element of duration tuple)
    block_start_time = seq.duration()[0] if len(seq.block_events) > 0 else 0

    rf, gz, gz_reph = make_rf_slice(flip_angle)
    rf.phase_offset = rf_phase

    gx_pre, gx, adc = make_epi_readout()

    seq.add_block(rf, gz)
    seq.add_block(gz_reph)
    seq.add_block(gx_pre)

    for i in range(72):  # 72 lines to acquire
        if i % 2 == 0:
            seq.add_block(gx, adc)
        else:
            gx.amplitude = -gx.amplitude
            seq.add_block(gx, adc)

        if i < 71:
            gy_blip_time = np.ceil(0.2e-3 / grad_raster) * grad_raster
            gy_blip = make_trapezoid(channel='y',
                                     system=system,
                                     area=delta_k,
                                     duration=gy_blip_time)
            seq.add_block(gy_blip)

    # Calculate total block duration (difference between current and start time)
    return seq.duration()[0] - block_start_time if len(seq.block_events) > 0 else 0


# Main sequence creation
rf_phase = 0
rf_spoiling_inc = 117

# Convert TR times from ms to seconds
tr_times = tr_times * 1e-3  # Convert from ms to seconds

for flip_angle, tr in zip(fa_angles, tr_times):
    # Add EPI block and get its duration
    block_duration = add_epi_block(seq, flip_angle, rf_phase)

    # Calculate next RF spoiling phase
    rf_phase = rf_phase + rf_spoiling_inc
    rf_spoiling_inc = rf_spoiling_inc + 117
    rf_phase = rf_phase % 360

    # Add delay to match TR if needed
    if block_duration < tr:
        delay_time = tr - block_duration
        seq.add_block(make_delay(delay_time))

# Write the sequence
seq.write("mrf_epi.seq")

# Plot sequence
seq.plot()
plt.show()


# Simulation and reconstruction
signal = mr0.util.simulate_2d(seq)

# Calculate actual acquired lines
lines_acquired = int(n * 9/16 * 1/R)

# Reshape signal considering partial k-space and acceleration
kspace_adc = torch.reshape((signal), (lines_acquired, n, len(tr_times))).clone()
kspace_adc = kspace_adc.transpose(0, 1)
kspace = torch.zeros((n, n, len(tr_times)), dtype=torch.complex64)

# Fill acquired k-space lines into full k-space matrix
acquired_line_indices = range(0, n * 9//16, R)
for time_step in range(len(tr_times)):
    for idx, k_idx in enumerate(acquired_line_indices):
        if idx % 2 == 0:
            kspace[:, k_idx, time_step] = kspace_adc[:, idx, time_step]
        else:
            kspace[:, k_idx, time_step] = torch.flip(kspace_adc[:, idx, time_step], [0])


# Convert to numpy for POCS reconstruction
kspace = kspace.numpy()
from partial_fourier_recon import pocs_pf
for time_step in range(kspace.shape[-1]):
    # add one dimension to k-space for POCS
    kspace = np.expand_dims(kspace, axis=0)
    pocs_pf(np.expand_dims(kspace[:, :, time_step], axis=0), 10)
    kspace[:, :, time_step] = pocs_pf(kspace[:, :, time_step], 10)
    # Convert back to torch and prepare for FFT
    kspace_slice = torch.tensor(kspace[:, :, time_step])
    kspace_slice = kspace_slice.squeeze()
    kspace_slice = torch.transpose(kspace_slice, 0, 1)

    # FFT reconstruction
    spectrum = torch.fft.fftshift(kspace_slice)
    space = torch.fft.fft2(spectrum)
    space = torch.fft.ifftshift(space)

    # Plotting
    fig = plt.figure(figsize=(10,2))

    plt.subplot(141)
    plt.title(f'k-space (9/16, R=2)_{time_step}')
    plt.imshow(np.abs(kspace), cmap='gray', aspect='auto')

    plt.subplot(142)
    plt.title(f'log. k-space_{time_step}')
    plt.imshow(np.log(np.abs(kspace) + 1) + 100, cmap='gray')

    plt.subplot(143)
    plt.title(f'FFT-magnitude_{time_step}')
    plt.imshow(np.abs(space.numpy()), cmap='gray')
    plt.colorbar()

    plt.subplot(144)
    plt.title(f'FFT-phase_{time_step}')
    plt.imshow(np.angle(space.numpy()), cmap='gray')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


