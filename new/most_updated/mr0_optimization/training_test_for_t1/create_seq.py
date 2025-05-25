import MRzeroCore as mr0
import pypulseq as pp
import numpy as np
import torch
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower

experiment_id = 'FLASH_2D_Fit'

# %% S1. SETUP sys

# choose the scanner limits
system = pp.Opts(max_grad=28,grad_unit='mT/m',max_slew=150,slew_unit='T/m/s',
                 rf_ringdown_time=20e-6,rf_dead_time=100e-6,adc_dead_time=20e-6,grad_raster_time=50*10e-6)

TI = [0.1,0.5,1.0,5.0]
n_TI = len(TI)
# Define FOV and resolution
fov = 200e-3
slice_thickness = 8e-3
Nread = 128    # frequency encoding steps/samples
Nphase = 128   # phase encoding steps/samples

##linear reordering
phenc = np.arange(-Nphase // 2, Nphase // 2, 1) / fov
# permvec = np.arange(0, Nphase, 1)
## centric reordering
permvec = sorted(np.arange(len(phenc)), key=lambda x: abs(len(phenc) // 2 - x))
## random reordering
#perm =np.arange(0, Nphase, 1);  permvec = np.random.permutation(perm)

# %% S2. DEFINE the sequence
def create_inv_rec(TI):
  seq = pp.Sequence()
  # Define rf events
  rf1, _, _ = pp.make_sinc_pulse(
      flip_angle=10 * np.pi / 180, duration=1e-3,
      slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
      system=system, return_gz=True
  )

  rf_inv = pp.make_block_pulse(flip_angle=180 * np.pi / 180, duration=1e-3, system=system)


  # rf1 = pp.make_block_pulse(flip_angle=90 * np.pi / 180, duration=1e-3, system=system)

  # Define other gradients and ADC events
  gx = pp.make_trapezoid(channel='x', flat_area=Nread / fov, flat_time=4e-3, system=system)
  adc = pp.make_adc(num_samples=Nread, duration=4e-3, phase_offset=0 * np.pi/180, delay=gx.rise_time, system=system)
  gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)
  gx_spoil = pp.make_trapezoid(channel='x', area=1.5 * gx.area, duration=2e-3, system=system)
  gy_spoil = pp.make_trapezoid(channel='y', area=1.5 * gx.area, duration=2e-3, system=system)

  rf_phase = 0
  rf_inc = 0
  rf_spoiling_inc = 117

  # ======
  # CONSTRUCT SEQUENCE
  # ======

  phenc_centr = phenc[permvec]
  for t in TI:
    seq.add_block(rf_inv)
    seq.add_block(pp.make_delay(t))
    seq.add_block(gx_spoil, gy_spoil)
    for ii in range(0, Nphase):  # e.g. -64:63

        rf1.phase_offset = rf_phase / 180 * np.pi   # set current rf phase

        adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]   # increase increment
        # increment additional pahse
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

        seq.add_block(rf1)
        seq.add_block(pp.make_delay(0.005))
        gp = pp.make_trapezoid(channel='y', area=phenc_centr[ii], duration=1e-3, system=system)
        seq.add_block(gx_pre, gp)
        seq.add_block(adc, gx)
        gp = pp.make_trapezoid(channel='y', area=-phenc_centr[ii], duration=1e-3, system=system)
        seq.add_block(gx_spoil, gp)
    seq.add_block(pp.make_delay(10))
  return seq

seq = create_inv_rec(TI)
ok, error_report = seq.check_timing()
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]
seq.write("FLASH_2D_Fit.seq")