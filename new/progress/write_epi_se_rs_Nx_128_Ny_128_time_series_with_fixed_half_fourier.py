import math

import numpy as np

import pypulseq as pp


def main(plot: bool = False, write_seq: bool = False,
         seq_filename: str = '6.5.25_epi_time_series_with_inversion_spoiler_gradient_half_fourier.seq'):
    # ======
    # SETUP
    # ======
    fov = 220e-3  # Define FOV and resolution
    Nx = 128
    Ny = 128
    slice_thickness = 3e-3  # Slice thickness
    n_slices = 1
    TE = 0.15
    pe_enable = 1  # Flag to quickly disable phase encoding (1/0) as needed for the delay calibration
    ro_os = 1  # Oversampling factor
    readout_time = 2 * 4.2e-4  # Readout bandwidth
    # Partial Fourier factor: 1: full sampling; 0.5: sample from -kmax to 0
    part_fourier_factor = 1
    t_RF_ex = 2e-3
    t_RF_ref = 2e-3
    spoil_factor = 1.5  # Spoiling gradient around the pi-pulse (rf180)
    flip_angles = [
        15.000000, 15.000000, 15.000000, 15.000000, 21.000000, 41.000000, 74.000000, 90.000000,
        90.000000, 90.000000, 90.000000, 15.000000, 15.000000, 15.000000, 15.000000, 15.000000,
        15.000000, 15.000000, 15.000000, 21.000000, 41.000000, 74.000000, 90.000000, 90.000000,
        90.000000, 90.000000, 15.000000, 15.000000, 15.000000, 15.000000, 15.000000, 15.000000,
        15.000000, 15.000000, 21.000000, 41.000000, 74.000000, 90.000000, 90.000000, 90.000000,
        90.000000, 15.000000, 15.000000, 15.000000, 15.000000, 15.000000, 15.000000, 15.000000,
        15.000000, 21.000000
    ]
    tr_values_ms = [
        200, 200, 200, 138, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        159, 200, 200, 200, 138, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        159, 200, 200, 200, 138, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        159, 200, 200, 200, 138, 75
    ]
    flip_angles = flip_angles[:2]
    tr_values_ms = tr_values_ms[:2]
    # Convert from milliseconds to seconds for the sequence timing
    tr_values = [tr_ms / 1000.0 for tr_ms in tr_values_ms]

    steps_number = len(flip_angles)
    # Set system limits
    assert len(flip_angles) == len(tr_values)

    system = pp.Opts(
        max_grad=60,
        grad_unit='mT/m',
        max_slew=150,
        slew_unit='T/m/s',
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
    )

    seq = pp.Sequence(system)  # Create a new sequence object

    # ======
    # CREATE EVENTS
    # ======
    # Create fat-sat pulse
    B0 = 2.89
    sat_ppm = -3.45
    sat_freq = sat_ppm * 1e-6 * B0 * system.gamma
    rf_fs = pp.make_gauss_pulse(
        flip_angle=110 * np.pi / 180,
        system=system,
        duration=8e-3,
        bandwidth=np.abs(sat_freq),
        freq_offset=sat_freq,
        delay=system.rf_dead_time,
    )
    gz_fs = pp.make_trapezoid(channel='z', system=system, delay=pp.calc_duration(rf_fs), area=1 / 1e-4)

    tr_spoil_factor = 2.0  # Adjust this value as needed for sufficient dephasing

    rf_pulses = []
    for flip_angle in flip_angles:
        # Create 90 degree slice selection pulse and gradient
        rf, gz, gz_reph = pp.make_sinc_pulse(
            flip_angle=flip_angle * (np.pi / 180),
            system=system,
            duration=t_RF_ex,
            slice_thickness=slice_thickness,
            apodization=0.5,
            time_bw_product=4,
            return_gz=True,
            delay=system.rf_dead_time,
        )
        rf_pulses.append(rf)

    tr_spoiler = pp.make_trapezoid(
        channel='z',  # Apply along slice selection direction
        system=system,
        area=tr_spoil_factor * gz.area,  # Scale based on slice selection gradient
        duration=3e-3  # Typical duration, adjust as needed
    )

    # Create 90 degree slice refocusing pulse and gradients
    rf180, gz180, _ = pp.make_sinc_pulse(
        flip_angle=np.pi,
        system=system,
        duration=t_RF_ref,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        phase_offset=np.pi / 2,
        use='refocusing',
        return_gz=True,
        delay=system.rf_dead_time,
    )
    _, gzr1_t, gzr1_a = pp.make_extended_trapezoid_area(
        channel='z',
        grad_start=0,
        grad_end=gz180.amplitude,
        area=spoil_factor * gz.area,
        system=system,
    )
    _, gzr2_t, gzr2_a = pp.make_extended_trapezoid_area(
        channel='z',
        grad_start=gz180.amplitude,
        grad_end=0,
        area=-gz_reph.area + spoil_factor * gz.area,
        system=system,
    )
    if gz180.delay > (gzr1_t[3] - gz180.rise_time):
        gz180.delay -= gzr1_t[3] - gz180.rise_time
    else:
        rf180.delay += (gzr1_t[3] - gz180.rise_time) - gz180.delay
    gz180n = pp.make_extended_trapezoid(
        channel='z',
        system=system,
        times=np.array([*gzr1_t, *gzr1_t[3] + gz180.flat_time + gzr2_t]) + gz180.delay,
        amplitudes=np.array([*gzr1_a, *gzr2_a]),
    )

    # Define the output trigger to play out with every slice excitation
    trig = pp.make_digital_output_pulse(channel='osc0', duration=100e-6)

    # Define other gradients and ADC events
    delta_k = 1 / fov
    k_width = Nx * delta_k

    # Phase blip in shortest possible time
    # Round up the duration to 2x gradient raster time
    blip_duration = np.ceil(2 * np.sqrt(delta_k / system.max_slew) / 10e-6 / 2) * 10e-6 * 2
    # Use negative blips to save one k-space line on our way to center of k-space
    gy = pp.make_trapezoid(channel='y', system=system, area=-delta_k, duration=blip_duration)

    # Readout gradient is a truncated trapezoid with dead times at the beginning and at the end each equal to a half of
    # blip duration. The area between the blips should be defined by k_width. We do a two-step calculation: we first
    # increase the area assuming maximum slew rate and then scale down the amplitude to fix the area
    extra_area = blip_duration / 2 * blip_duration / 2 * system.max_slew
    gx = pp.make_trapezoid(
        channel='x',
        system=system,
        area=k_width + extra_area,
        duration=readout_time + blip_duration,
    )
    actual_area = gx.area - gx.amplitude / gx.rise_time * blip_duration / 2 * blip_duration / 2 / 2
    actual_area -= gx.amplitude / gx.fall_time * blip_duration / 2 * blip_duration / 2 / 2
    gx.amplitude = gx.amplitude / actual_area * k_width
    gx.area = gx.amplitude * (gx.flat_time + gx.rise_time / 2 + gx.fall_time / 2)
    gx.flat_area = gx.amplitude * gx.flat_time

    # Calculate ADC
    # We use ramp sampling, so we have to calculate the dwell time and the number of samples, which will be quite
    # different from Nx and readout_time/Nx, respectively.
    adc_dwell_nyquist = delta_k / gx.amplitude / ro_os
    # Round-down dwell time to 100 ns
    adc_dwell = math.floor(adc_dwell_nyquist * 1e7) * 1e-7
    # Number of samples on Siemens needs to be divisible by 4
    adc_samples = math.floor(readout_time / adc_dwell / 4) * 4
    adc = pp.make_adc(num_samples=adc_samples, dwell=adc_dwell, delay=blip_duration / 2)
    # Realign the ADC with respect to the gradient
    # Supposedly Siemens samples at center of dwell period
    time_to_center = adc_dwell * ((adc_samples - 1) / 2 + 0.5)
    # Adjust delay to align the trajectory with the gradient. We have to align the delay to 1us
    adc.delay = round((gx.rise_time + gx.flat_time / 2 - time_to_center) * 1e6) * 1e-6
    # This rounding actually makes the sampling points on odd and even readouts to appear misaligned. However, on the
    # real hardware this misalignment is much stronger anyways due to the gradient delays

    # Split the blip into two halves and produce a combined synthetic gradient
    gy_parts = pp.split_gradient_at(grad=gy, time_point=blip_duration / 2, system=system)
    gy_blipup, gy_blipdown, _ = pp.align(right=gy_parts[0], left=[gy_parts[1], gx])
    gy_blipdownup = pp.add_gradients((gy_blipdown, gy_blipup), system=system)

    # pe_enable support
    gy_blipup.waveform = gy_blipup.waveform * pe_enable
    gy_blipdown.waveform = gy_blipdown.waveform * pe_enable
    gy_blipdownup.waveform = gy_blipdownup.waveform * pe_enable

    # Phase encoding and partial Fourier
    # MODIFIED: Always start from maximum negative k-space (-kmax)
    # For full k-space acquisition (part_fourier_factor=1):
    # - Ny_pre = Ny/2 - 1 lines before ky=0 (excluding center line)
    # - Ny_post = Ny/2 + 1 lines after and including ky=0
    # For partial Fourier, we keep Ny_pre the same but adjust Ny_post based on factor
    
    # Ensure part_fourier_factor is at least 0.5
    assert part_fourier_factor >= 0.5, "Partial Fourier factor must be at least 0.5"
    
    # Calculate number of lines before k-space center (excluding center)
    Ny_pre = round(Ny / 2 - 1)
    
    # Calculate number of lines after k-space center (including center)
    # For part_fourier_factor=1, this equals Ny/2 + 1
    # For part_fourier_factor=0.5, this would equal 1 (just the center line)
    Ny_post = max(1, round(part_fourier_factor * Ny - Ny_pre))
    
    # Total number of measured lines
    Ny_meas = Ny_pre + Ny_post
    
    print(f"Sampling {Ny_meas} lines: {Ny_pre} lines before center and {Ny_post} lines after/including center")

    # Pre-phasing gradients - prepare to start at -kmax
    gx_pre = pp.make_trapezoid(channel='x', system=system, area=-gx.area / 2)
    
    # Always prephase to the most negative k-space line
    gy_pre = pp.make_trapezoid(channel='y', system=system, area=Ny_pre * delta_k)

    gx_pre, gy_pre = pp.align(right=gx_pre, left=gy_pre)
    # Relax the PE prephaser to reduce stimulation
    gy_pre = pp.make_trapezoid('y', system=system, area=gy_pre.area, duration=pp.calc_duration(gx_pre, gy_pre))
    gy_pre.amplitude = gy_pre.amplitude * pe_enable

    # Calculate delay times
    duration_to_center = (Ny_pre + 0.5) * pp.calc_duration(gx)
    rf_center_incl_delay = rf.delay + pp.calc_rf_center(rf)[0]
    rf180_center_incl_delay = rf180.delay + pp.calc_rf_center(rf180)[0]
    delay_TE1 = (
            math.ceil(
                (TE / 2 - pp.calc_duration(rf, gz) + rf_center_incl_delay - rf180_center_incl_delay)
                / system.grad_raster_time
            )
            * system.grad_raster_time
    )
    delay_TE2 = (
            math.ceil(
                (TE / 2 - pp.calc_duration(rf180, gz180n) + rf180_center_incl_delay - duration_to_center)
                / system.grad_raster_time
            )
            * system.grad_raster_time
    )
    assert delay_TE1 >= 0
    # Now we merge slice refocusing, TE delay and pre-phasers into a single block
    delay_TE2 = delay_TE2 + pp.calc_duration(rf180, gz180n)
    gx_pre.delay = 0
    gx_pre.delay = delay_TE2 - pp.calc_duration(gx_pre)
    assert gx_pre.delay >= pp.calc_duration(rf180)  # gx_pre may not overlap with the RF
    gy_pre.delay = pp.calc_duration(rf180)
    # gy_pre may not shift the timing
    assert pp.calc_duration(gy_pre) <= pp.calc_duration(gx_pre)

    # ======
    # CONSTRUCT SEQUENCE
    # ======
    for s in range(n_slices):
        # 1. INITIAL ADIABATIC INVERSION PULSE (once per slice)
        rf_inv, gz_inv, gzr_inv = pp.make_adiabatic_pulse(
            pulse_type='hypsec',
            duration=4e-3,
            delay=system.rf_dead_time,
            system=system,
            use='inversion',
            slice_thickness=slice_thickness,
            return_gz=True
        )
        # Set slice position
        rf_inv.freq_offset = gz_inv.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
        # Add the pulse and gradient
        seq.add_block(rf_inv, gz_inv)
        # Add the rephasing gradient
        seq.add_block(gzr_inv)

        # 2. Add a delay after inversion (typically 10-30ms)
        seq.add_block(pp.make_delay(40e-3))

        # 3. MULTIPLE EXCITATIONS FOR THIS SLICE
        for t in range(steps_number):
            # Continue with your existing sequence...
            seq.add_block(rf_fs, gz_fs)
            rf.freq_offset = gz.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
            rf180.freq_offset = gz180.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
            seq.add_block(rf_pulses[t], gz, trig)
            seq.add_block(pp.make_delay(delay_TE1))
            seq.add_block(rf180, gz180n, pp.make_delay(delay_TE2), gx_pre, gy_pre)

            # EPI readout
            for i in range(1, Ny_meas + 1):
                if i == 1:
                    seq.add_block(gx, gy_blipup, adc)
                elif i == Ny_meas:
                    seq.add_block(gx, gy_blipdown, adc)
                else:
                    seq.add_block(gx, gy_blipdownup, adc)
                gx.amplitude = -gx.amplitude

            # End-of-TR spoiler
            seq.add_block(tr_spoiler)

            current_seq_blocks_duration = pp.calc_duration(rf_pulses[t], gz)  # Excitation
            current_seq_blocks_duration += delay_TE1  # First TE delay
            current_seq_blocks_duration += pp.calc_duration(rf180, gz180n, gx_pre, gy_pre)  # Refocusing + prep
            current_seq_blocks_duration += Ny_meas * pp.calc_duration(gx, adc)  # EPI readout
            current_seq_blocks_duration += pp.calc_duration(tr_spoiler)  # End spoiler
            print(f"TR {t+1}: Total sequence duration: {current_seq_blocks_duration:.6f} s")

            desired_tr = tr_values[t]  # Get the TR for this repetition in seconds
            additional_delay = desired_tr - current_seq_blocks_duration

            # Add a delay to achieve the desired TR if needed
            if additional_delay > 0:
                seq.add_block(pp.make_delay(additional_delay))
                print(f"Adding delay of {additional_delay:.6f} s to reach desired TR of {desired_tr:.6f} s")
            else:
                print(
                    f"Warning: TR {t+1} ({tr_values[t] * 1000:.1f} ms) is too short! Minimum possible TR is {current_seq_blocks_duration * 1000:.1f} ms")

    # Check whether the timing of the sequence is correct
    ok, error_report = seq.check_timing()
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]

    # Very optional slow step, but useful for testing during development e.g. for the real TE, TR or for staying within
    # slew-rate limits
    rep = seq.test_report()
    print(rep)

    # ======
    # VISUALIZATION
    # ======
    if plot:
        seq.plot()

    # =========
    # WRITE .SEQ
    # =========
    if write_seq:
        # Prepare the sequence output for the scanner
        seq.set_definition(key='FOV', value=[fov, fov, slice_thickness])
        seq.set_definition(key='Name', value='epi')

        seq.write(seq_filename)

    return seq


if __name__ == '__main__':
    main(plot=True, write_seq=True)
