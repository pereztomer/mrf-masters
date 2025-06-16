import math
import numpy as np
import pypulseq as pp
import matplotlib.pyplot as plt

def main():
    """
    Experimental high-performance EPI sequence with custom TE control
    Uses split gradients to overlap blips with readout gradients and ramp sampling
    """
    
    # Set system limits
    sys = pp.Opts(
        max_grad=60, grad_unit='mT/m',
        max_slew=100, slew_unit='T/m/s',
        rf_ringdown_time=20e-6, rf_dead_time=100e-6,
        adc_dead_time=10e-6, B0=2.89  # Siemens 3T
    )

    seq = pp.Sequence(sys)  # Create a new sequence object
    fov = 256e-3
    Nx = 128
    Ny = Nx  # Define FOV and resolution
    thickness = 4e-3  # slice thickness in mm
    slice_gap = 1e-3  # slice gap in mm
    Nslices = 1

    # ===== CUSTOM TE CONTROL =====
    # Set your desired TE here (in seconds)
    TE = 30e-3  # 30 ms - change this to your desired value
    # Minimum TE will be calculated and enforced if needed

    pe_enable = 1  # flag to quickly disable phase encoding (1/0)
    ro_os = 1  # oversampling factor
    readout_time = 15e-4  # controls readout bandwidth
    part_fourier_factor = 1  # partial Fourier factor: 1: full sampling 0: start with ky=0

    # Create fat-sat pulse
    sat_ppm = -3.45
    sat_freq = sat_ppm * 1e-6 * sys.B0 * sys.gamma
    rf_fs = pp.make_gauss_pulse(
        flip_angle=110 * np.pi / 180, system=sys, duration=8e-3,
        bandwidth=abs(sat_freq), freq_offset=sat_freq, use='saturation'
    )
    rf_fs.phase_offset = -2 * np.pi * rf_fs.freq_offset * pp.calc_rf_center(rf_fs)[0]
    gz_fs = pp.make_trapezoid(channel='z', system=sys, delay=pp.calc_duration(rf_fs), area=0.1 / 1e-4)

    # Create 90 degree slice selection pulse and gradient
    rf, gz, gz_reph = pp.make_sinc_pulse(
        flip_angle=np.pi / 2, system=sys, duration=2e-3,
        slice_thickness=thickness, apodization=0.42, time_bw_product=4,
        use='excitation', return_gz=True
    )

    # Define output trigger
    trig = pp.make_digital_output_pulse('osc0', duration=100e-6)

    # Define other gradients and ADC events
    delta_k = 1 / fov
    k_width = Nx * delta_k

    # Phase blip in shortest possible time
    blip_dur = math.ceil(2 * math.sqrt(delta_k / sys.max_slew) / 10e-6 / 2) * 10e-6 * 2
    gy = pp.make_trapezoid(channel='y', system=sys, area=blip_dur)

    # Readout gradient design
    extra_area = blip_dur / 2 * blip_dur / 2 * sys.max_slew
    gx = pp.make_trapezoid('x', system=sys, area=k_width + extra_area, duration=readout_time + blip_dur)
    
    # Adjust gradient amplitude for actual area calculation
    actual_area = (gx.area - gx.amplitude / gx.rise_time * blip_dur / 2 * blip_dur / 2 / 2 -
                   gx.amplitude / gx.fall_time * blip_dur / 2 * blip_dur / 2 / 2)
    gx.amplitude = gx.amplitude / actual_area * k_width
    gx.area = gx.amplitude * (gx.flat_time + gx.rise_time / 2 + gx.fall_time / 2)
    gx.flat_area = gx.amplitude * gx.flat_time

    # Calculate ADC with ramp sampling
    adc_dwell_nyquist = delta_k / gx.amplitude / ro_os
    adc_dwell = math.floor(adc_dwell_nyquist * 1e7) * 1e-7
    adc_samples = math.floor(readout_time / adc_dwell / 4) * 4
    adc = pp.make_adc(num_samples=adc_samples, dwell=adc_dwell, delay=blip_dur / 2)

    # Realign ADC with gradient
    time_to_center = adc.dwell * ((adc_samples - 1) / 2 + 0.5)
    adc.delay = round((gx.rise_time + gx.flat_time / 2 - time_to_center) * 1e6) * 1e-6

    # Split blip gradients
    gy_parts = pp.split_gradient_at(gy, blip_dur / 2, sys)
    gy_blipup, gy_blipdown, _ = pp.align(right=gy_parts[0], left=[gy_parts[1], gx])

    gy_blipdownup = pp.add_gradients((gy_blipdown, gy_blipup), system=sys)

    # PE enable support
    if pe_enable == 0:
        gy_blipup.waveform = gy_blipup.waveform * 0
        gy_blipdown.waveform = gy_blipdown.waveform * 0
        gy_blipdownup.waveform = gy_blipdownup.waveform * 0

    # Phase encoding and partial Fourier
    Ny_pre = round(part_fourier_factor * Ny / 2 - 1)
    Ny_post = round(Ny / 2 + 1)
    Ny_meas = Ny_pre + Ny_post

    # Pre-phasing gradients
    gx_pre = pp.make_trapezoid('x', system=sys, area=-gx.area / 2)
    gy_pre = pp.make_trapezoid('y', system=sys, area=Ny_pre * delta_k)
    # gx_pre, gy_pre, gz_reph = pp.align('right', gx_pre, 'left', gy_pre, gz_reph)
    gx_pre, gy_pre = pp.align(right=gx_pre, left=gy_pre)
    gy_pre = pp.make_trapezoid('y', system=sys, area=gy_pre.area,
                               duration=pp.calc_duration(gx_pre, gy_pre, gz_reph))
    if pe_enable == 0:
        gy_pre.amplitude = gy_pre.amplitude * 0

    # Slice positions
    slice_positions = (thickness + slice_gap) * (np.arange(Nslices) - (Nslices - 1) / 2)
    # Reorder slices (odd first, then even)
    slice_positions = np.concatenate([slice_positions[::2], slice_positions[1::2]])

    # ===== IMPROVED TE TIMING CALCULATION =====
    # Calculate timing components more accurately
    rf_center_time = pp.calc_rf_center(rf)[0]  # Time to RF center (excitation)
    prephaser_duration = pp.calc_duration(gx_pre, gy_pre, gz_reph)
    first_readout_duration = pp.calc_duration(gx, gy_blipup, adc)

    # Time from RF center to center of k-space (first ADC sample of central k-space line)
    time_to_kspace_center = prephaser_duration + (Ny_pre + 0.5) * pp.calc_duration(gx, gy_blipdownup, adc)

    # Calculate minimum achievable TE
    min_TE = rf_center_time + time_to_kspace_center

    print('=== TE TIMING ANALYSIS ===')
    print(f'RF center time: {rf_center_time:.6f} s')
    print(f'Time to k-space center: {time_to_kspace_center:.6f} s')
    print(f'Minimum achievable TE: {min_TE:.6f} s')
    print(f'Requested TE: {TE:.6f} s')

    # Check if requested TE is achievable
    if TE < min_TE:
        print(f'WARNING: Requested TE ({TE:.6f} s) is shorter than minimum TE ({min_TE:.6f} s)')
        print(f'Setting TE to minimum value: {min_TE:.6f} s')
        TE = min_TE
    else:
        print(f'TE is achievable. Additional delay needed: {(TE - min_TE):.6f} s')

    # Define sequence blocks
    for s in range(Nslices):
        # Fat sat
        seq.add_block(rf_fs, gz_fs)
        
        # Excitation
        rf.freq_offset = gz.amplitude * slice_positions[s]
        rf.phase_offset = -2 * np.pi * rf.freq_offset * pp.calc_rf_center(rf)[0]
        seq.add_block(rf, gz, trig)
        
        # ===== TE DELAY IMPLEMENTATION =====
        # Calculate delay needed to achieve desired TE
        delay_needed = TE - min_TE
        
        if delay_needed > 0:
            # Add delay block with proper rounding to gradient raster
            delay_block = pp.make_delay(delay_needed)
            seq.add_block(delay_block)
            print(f'Added TE delay: {delay_needed:.6f} s')
        
        # Pre-phasing
        seq.add_block(gx_pre, gy_pre, gz_reph)
        
        # EPI readout train
        for i in range(Ny_meas):
            if i == 0:
                seq.add_block(gx, gy_blipup, adc)
            elif i == Ny_meas - 1:
                seq.add_block(gx, gy_blipdown, adc)
            else:
                seq.add_block(gx, gy_blipdownup, adc)
            gx.amplitude = -gx.amplitude  # Reverse polarity of read gradient

    # Verify timing
    ok, error_report = seq.check_timing()

    if ok:
        print('\nTiming check passed successfully')
    else:
        print('\nTiming check failed! Error listing follows:')
        print(error_report)

    # Calculate and display actual TE achieved
    try:
        ktraj_adc, ktraj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

        if len(t_excitation) > 0 and len(t_adc) > 0:
            # Find the ADC samples corresponding to the central k-space line
            center_line_start_idx = Ny_pre * adc_samples
            center_line_center_idx = center_line_start_idx + adc_samples // 2
            
            if len(t_adc) > center_line_center_idx:
                # Time from RF center to center of k-space
                actual_TE = t_adc[center_line_center_idx] - (t_excitation[0] + rf_center_time)
                
                print('\n=== FINAL TE VERIFICATION ===')
                print(f'RF excitation time: {t_excitation[0]:.6f} s')
                print(f'RF center time: {(t_excitation[0] + rf_center_time):.6f} s')
                print(f'K-space center acquisition time: {t_adc[center_line_center_idx]:.6f} s')
                print(f'Actual achieved TE: {actual_TE:.6f} s')
                print(f'Target TE: {TE:.6f} s')
                print(f'TE error: {(actual_TE - TE):.6f} s')
            else:
                print('\n=== TE VERIFICATION ===')
                print('Cannot verify TE - insufficient ADC samples in trajectory')
                print(f'Expected center sample index: {center_line_center_idx}, Available samples: {len(t_adc)}')
        else:
            print('\n=== TE VERIFICATION ===')
            print('Cannot verify TE - trajectory calculation failed')
    except Exception as e:
        print(f'\n=== TE VERIFICATION ===')
        print(f'Cannot verify TE - trajectory calculation error: {e}')

    # Plot sequence
    seq.plot()
    seq.plot(time_disp='us', show_blocks=True, time_range=[0, 25e-3])

    # Set sequence definitions
    seq.set_definition('Name', 'epi_custom_te')
    seq.set_definition('FOV', [fov, fov, max(slice_positions) - min(slice_positions) + thickness])
    seq.set_definition('ReceiverGainHigh', 1)

    # Write sequence file
    seq.write('epi_custom_te.seq')
    print('\nSequence written to: epi_custom_te.seq')

    return seq

if __name__ == '__main__':
    main()