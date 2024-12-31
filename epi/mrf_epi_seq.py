import numpy as np
import MRzeroCore as mr0
import pypulseq as pp
import torch
import matplotlib.pyplot as plt

# Setup system limits
system = pp.Opts(
    max_grad=28, grad_unit='mT/m',
    max_slew=150, slew_unit='T/m/s',
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=20e-6,
    grad_raster_time=10e-6
)

# Initialize sequence
seq = pp.Sequence()

# Define FOV and resolution
fov = 240e-3
slice_thickness = 8e-3
Nread = 64
Nphase = 64
partial_factor = 9 / 16
R = 2  # Acceleration factor

# Calculate number of lines to acquire with partial Fourier and acceleration
lines_to_acquire = int(Nphase * partial_factor / R)  # Divided by R for acceleration

# Create 180° inversion pulse
rf_inv = pp.make_block_pulse(
    flip_angle=180 * np.pi / 180,
    duration=1e-3,
    system=system
)

# Create excitation pulse
rf_exc, gz, _ = pp.make_sinc_pulse(
    flip_angle=15 * np.pi / 180,
    duration=1e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system,
    return_gz=True
)

# Calculate gradient and ADC requirements for single-shot
readout_time = 0.25e-3
gx_pos = pp.make_trapezoid(
    channel='x',
    flat_area=Nread / fov,
    flat_time=readout_time,
    system=system
)

# Create negative gradient explicitly
gx_neg = pp.make_trapezoid(
    channel='x',
    flat_area=-Nread / fov,  # Negative area
    flat_time=readout_time,
    system=system
)

# ADC event
adc = pp.make_adc(
    num_samples=Nread,
    duration=readout_time,
    delay=gx_pos.rise_time,
    system=system
)

# Calculate blip size for accelerated trajectory
blip_area = R / fov
gp_blip = pp.make_trapezoid(
    channel='y',
    area=blip_area,
    duration=0.1e-3,
    system=system
)

# Sequence construction
# 1. Inversion pulse
seq.add_block(rf_inv)

# 2. TI delay
seq.add_block(pp.make_delay(0.040))  # 40ms TI

# 3. Excitation
seq.add_block(rf_exc, gz)

# 4. Prephasing
gp_pre = pp.make_trapezoid(
    channel='y',
    area=-lines_to_acquire * R * blip_area / 2,
    duration=1e-3,
    system=system
)
gx_pre = pp.make_trapezoid(
    channel='x',
    area=-gx_pos.area / 2,
    duration=1e-3,
    system=system
)
seq.add_block(gx_pre, gp_pre)

# 5. EPI readout
for line in range(lines_to_acquire):
    if line % 2 == 0:
        seq.add_block(gx_pos, adc)
    else:
        seq.add_block(gx_neg, adc)  # Using explicit negative gradient

    if line < lines_to_acquire - 1:  # No blip after last line
        seq.add_block(gp_blip)

# Plot sequence
seq.plot()
plt.show()

# Print sequence info
print(f"Sequence duration: {seq.duration()[0]:.2f} s")
print(f"Number of k-space lines: {lines_to_acquire} (of {Nphase} total)")
print(f"Acceleration factor: {R}")
print(f"Partial Fourier factor: {partial_factor}")


# Quick 2D brain phantom sim and plot
signal = mr0.util.simulate_2d(seq)
seq.plot(plot_now=False)
mr0.util.insert_signal_plot(seq=seq, signal=signal.numpy())
plt.show()

# MR IMAGE RECONSTRUCTION
fig = plt.figure(figsize=(10,2))

# Calculate actual acquired lines
lines_acquired = int(Nphase * partial_factor / R)

# Reshape signal considering partial k-space and acceleration
kspace_adc = torch.reshape((signal), (lines_acquired, Nread)).clone().t()
kspace = torch.zeros((Nread, Nphase), dtype=torch.complex64)

# Fill acquired k-space lines into full k-space matrix
# Consider both acceleration and partial Fourier
acquired_line_indices = range(0, Nphase * 9//16, R)  # R=2 steps for acceleration
for idx, k_idx in enumerate(acquired_line_indices):
    if idx % 2 == 0:
        kspace[:, k_idx] = kspace_adc[:, idx]
    else:
        kspace[:, k_idx] = torch.flip(kspace_adc[:, idx], [0])


import partial_fourier_recon
import scipy.io

# add to dims to kspace
# kspace = torch.unsqueeze(kspace, -1)
# kspace = torch.unsqueeze(kspace, -1)

# convert to numpy:
kspace = kspace.numpy()
from partial_fourier_recon import pocs_pf
kspace = pocs_pf(kspace, 10)
# kspFull_kxkycz, kspZpad_kxkycz = partial_fourier_recon.pf_recon_pocs_ms2d(kspace, 10)
# transpose the test data using torch
kspace = torch.tensor(kspace)
kspace = kspace.squeeze()

kspace = torch.transpose(kspace, 0, 1)

# drop first dim
# Apply fftshift before FFT
spectrum = torch.fft.fftshift(kspace)
space = torch.fft.fft2(spectrum)
space = torch.fft.ifftshift(space)

# convert to numpy
# space = space.numpy()
# Plotting
plt.subplot(141)
plt.title('k-space (9/16, R=2)')
plt.imshow(np.abs(kspace), cmap='gray', aspect='auto')
# mr0.util.imshow(np.abs(kspace.numpy()))

plt.subplot(142)
plt.title('log. k-space')
plt.imshow(np.log(np.abs(kspace) + 1) + 100, cmap='gray')
# mr0.util.imshow(np.log(np.abs(kspace.numpy()) + 1))  # Add 1 to avoid log(0)

plt.subplot(143)
plt.title('FFT-magnitude')
plt.imshow(np.abs(space.numpy()), cmap='gray')
# mr0.util.imshow(np.abs(space.numpy()))
plt.colorbar()

plt.subplot(144)
plt.title('FFT-phase')
plt.imshow(np.angle(space.numpy()), cmap='gray')
# mr0.util.imshow(np.angle(space.numpy()), vmin=-np.pi, vmax=np.pi)
plt.colorbar()

plt.tight_layout()
plt.show()

# Print acquisition info
print(f"Total k-space lines: {Nphase}")
print(f"Actually acquired lines: {lines_acquired}")
print(f"Acceleration factor: {R}")
print(f"Partial Fourier factor: {partial_factor}")