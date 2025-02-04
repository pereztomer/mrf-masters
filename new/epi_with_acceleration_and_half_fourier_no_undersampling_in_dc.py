import numpy as np
import MRzeroCore as mr0
import pypulseq as pp
import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

plot = True
write_seq = True
# ======
# SETUP
# ======
R = 4
seq = pp.Sequence()  # Create a new sequence object
# Define FOV and resolution
fov = 220e-3
Nx = 128
Ny = 128
TE = 10e-3
use_partial_fourier = True
partial_fourier_fraction = 9 / 16

seq_filename = f'epi_pypulseq_Nx_{Nx}_Ny_{Ny}_acceleration_R_{R}_half_fourier_{use_partial_fourier}.seq'

slice_thickness = 3e-3  # Slice thickness
n_slices = 1

# Set system limits
system = pp.Opts(
    max_grad=32,
    grad_unit='mT/m',
    max_slew=130,
    slew_unit='T/m/s',
    rf_ringdown_time=30e-6,
    rf_dead_time=100e-6,
)

# ======
# CREATE EVENTS
# ======
# Create 90 degree slice selection pulse and gradient
rf, gz, _ = pp.make_sinc_pulse(
    flip_angle=np.pi / 2,
    system=system,
    duration=3e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    return_gz=True,
)

# Define other gradients and ADC events
delta_k = 1 / fov
k_width = Nx * delta_k
dwell_time = 4e-6
readout_time = Nx * dwell_time
flat_time = np.ceil(readout_time * 1e5) * 1e-5  # round-up to the gradient raster
gx = pp.make_trapezoid(
    channel='x',
    system=system,
    amplitude=k_width / readout_time,
    flat_time=flat_time,
)
adc = pp.make_adc(
    num_samples=Nx,
    duration=readout_time,
    delay=gx.rise_time + flat_time / 2 - (readout_time - dwell_time) / 2,
)

# Pre-phasing gradients
pre_time = 32e-4
gx_pre = pp.make_trapezoid(channel='x', system=system, area=-gx.area / 2, duration=pre_time)
gz_reph = pp.make_trapezoid(channel='z', system=system, area=-gz.area / 2, duration=pre_time)
gy_pre = pp.make_trapezoid(channel='y', system=system, area=-Ny / 2 * delta_k, duration=pre_time)

print(f"pre y: {-Ny / 2 * delta_k}")
print("delta_k: ", delta_k)
print("Ny: ", Ny)
# Phase blip in the shortest possible time
dur = np.ceil(2 * np.sqrt(delta_k / system.max_slew) / 10e-6) * 10e-6
gy_acceleration_jump = pp.make_trapezoid(channel='y', system=system, area=delta_k*R, duration=dur*R)
gy_for_dc = pp.make_trapezoid(channel='y', system=system, area=delta_k, duration=dur)

# ======
# CONSTRUCT SEQUENCE
# ======
# Define sequence blocks
for s in range(n_slices):
    rf.freq_offset = gz.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
    seq.add_block(rf, gz)
    seq.add_block(gx_pre, gy_pre, gz_reph)
    i = 0
    for _ in range(Ny):
        seq.add_block(gx, adc)  # Read one line of k-space
        if i < int(Ny * 0.5 - 10) or i > int(Ny * 0.5 + 10):
            seq.add_block(gy_acceleration_jump)  # Phase blip
            i += R
        else:
            seq.add_block(gy_for_dc)  # Phase blip
            i += 1

        if i > Ny:
            break
        if use_partial_fourier and i > Ny * partial_fourier_fraction:
            break
        gx.amplitude = -gx.amplitude  # Reverse polarity of read gradient

ok, error_report = seq.check_timing()
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

# =========
# WRITE .SEQ
# =========
if write_seq:
    seq.write(seq_filename)
if plot:
    seq.plot()
