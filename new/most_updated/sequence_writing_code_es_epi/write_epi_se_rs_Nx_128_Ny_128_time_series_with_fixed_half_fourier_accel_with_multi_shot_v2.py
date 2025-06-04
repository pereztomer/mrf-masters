import math

import numpy as np

import pypulseq as pp
from datetime import date

# Get the current date
current_date = date.today()
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ====== ACCELERATION FACTOR ======
acceleration_factor = 3  # R=1 means fully sampled; R>1 means skip every R-th line


def main(plot: bool = False, write_seq: bool = False, seq_filename=f""):
    # ======
    # SETUP
    # ======
    fov = 220e-3  # Define FOV and resolution
    Nx = 192
    Ny = 192
    slice_thickness = 3e-3  # Slice thickness
    n_slices = 1
    TE = 0.2
    # TE = 18 / 1000
    pe_enable = 1  # Flag to quickly disable phase encoding (1/0) as needed for the delay calibration
    ro_os = 1  # Oversampling factor
    readout_time = 2 * 4.2e-4  # Readout bandwidth
    # Partial Fourier factor: 1: full sampling; 0.5: sample from -kmax to 0
    part_fourier_factor = 9 / 16
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

    # flip_angles = flip_angles[:1]
    # tr_values_ms = tr_values_ms[:1]
    if part_fourier_factor == 1:
        seq_filename = f"sequences/{current_date}_epi_multishot_ref_Nx{Nx}_Ny{Ny}_R{acceleration_factor}_repetitions_{len(tr_values_ms)}"
    else:
        seq_filename = f"sequences/{current_date}_epi_multishot_ref_Nx{Nx}_Ny{Ny}_R{acceleration_factor}_part_fourier_repetitions_{len(tr_values_ms)}"

    # Convert from milliseconds to seconds for the sequence timing
    tr_values = [tr_ms / 1000.0 for tr_ms in tr_values_ms]

    steps_number = len(flip_angles)
    # Set system limits
    assert len(flip_angles) == len(tr_values)

    system = pp.Opts(
        max_grad=60,
        grad_unit='mT/m',
        max_slew=100,
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
        freq_offset=sat_freq,
        delay=system.rf_dead_time,
        # Only specify bandwidth if you want to constrain it
        bandwidth=np.abs(sat_freq),
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

    # Create reference RF pulse (using first flip angle)
    rf_ref, gz_ref, gz_reph_ref = pp.make_sinc_pulse(
        flip_angle=flip_angles[0] * (np.pi / 180),
        system=system,
        duration=t_RF_ex,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        return_gz=True,
        delay=system.rf_dead_time,
    )

    # Create inversion pulse (to be reused)
    rf_inv, gz_inv, gzr_inv = pp.make_adiabatic_pulse(
        pulse_type='hypsec',
        duration=4e-3,
        delay=system.rf_dead_time,
        system=system,
        use='inversion',
        slice_thickness=slice_thickness,
        return_gz=True
    )

    tr_spoiler = pp.make_trapezoid(
        channel='z',  # Apply along slice selection direction
        system=system,
        area=tr_spoil_factor * gz.area,  # Scale based on slice selection gradient
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
    gy = pp.make_trapezoid(channel='y', system=system, area=-delta_k * acceleration_factor)
    blip_duration = pp.calc_duration(gy)

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

    adc_dwell_nyquist = delta_k / gx.amplitude / ro_os
    adc_dwell = math.floor(adc_dwell_nyquist * 1e7) * 1e-7
    adc_samples = math.floor(readout_time / adc_dwell / 4) * 4
    adc = pp.make_adc(num_samples=adc_samples, dwell=adc_dwell, delay=blip_duration / 2)
    time_to_center = adc_dwell * ((adc_samples - 1) / 2 + 0.5)
    adc.delay = round((gx.rise_time + gx.flat_time / 2 - time_to_center) * 1e6) * 1e-6

    gy_parts = pp.split_gradient_at(grad=gy, time_point=blip_duration / 2, system=system)
    gy_blipup, gy_blipdown, _ = pp.align(right=gy_parts[0], left=[gy_parts[1], gx])
    gy_blipdownup = pp.add_gradients((gy_blipdown, gy_blipup), system=system)

    gy_blipup.waveform = gy_blipup.waveform * pe_enable
    gy_blipdown.waveform = gy_blipdown.waveform * pe_enable
    gy_blipdownup.waveform = gy_blipdownup.waveform * pe_enable

    # Phase encoding and partial Fourier
    assert part_fourier_factor >= 0.5, "Partial Fourier factor must be at least 0.5"
    Ny_pre = round(Ny / 2 - 1)
    Ny_post = max(1, round(part_fourier_factor * Ny - Ny_pre))
    Ny_meas_full = Ny_pre + Ny_post
    pe_indices = list(range(1, Ny_meas_full + 1))
    pe_indices_accel = pe_indices[::acceleration_factor]
    Ny_meas = len(pe_indices_accel)
    print(
        f"Sampling {Ny_meas} lines (acceleration R={acceleration_factor}): {Ny_pre} lines before center and {Ny_post} lines after/including center (full would be {Ny_meas_full})")

    # For multi-shot reference: create separate pe_indices for each shot
    pe_indices_multishot = []
    for shot in range(acceleration_factor):
        shot_indices = pe_indices[shot::acceleration_factor]
        pe_indices_multishot.append(shot_indices)
        print(f"Shot {shot + 1} will acquire {len(shot_indices)} lines: {shot_indices[:5]}..." + (
            f" to {shot_indices[-1]}" if len(shot_indices) > 5 else ""))

    gx_pre = pp.make_trapezoid(channel='x', system=system, area=-gx.area / 2)
    gy_pre = pp.make_trapezoid(channel='y', system=system, area=Ny_pre * delta_k)
    gx_pre, gy_pre = pp.align(right=gx_pre, left=gy_pre)
    gy_pre = pp.make_trapezoid('y', system=system, area=gy_pre.area, duration=pp.calc_duration(gx_pre, gy_pre))
    gy_pre.amplitude = gy_pre.amplitude * pe_enable

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
    delay_TE2 = delay_TE2 + pp.calc_duration(rf180, gz180n)
    gx_pre.delay = 0
    gx_pre.delay = delay_TE2 - pp.calc_duration(gx_pre)
    assert gx_pre.delay >= pp.calc_duration(rf180)
    gy_pre.delay = pp.calc_duration(rf180)
    assert pp.calc_duration(gy_pre) <= pp.calc_duration(gx_pre)

    # ======
    # CONSTRUCT SEQUENCE
    # ======
    for s in range(n_slices):
        print(f"\n{'=' * 60}")
        print(f"SLICE {s + 1}/{n_slices}")
        print(f"{'=' * 60}")

        # ======
        # MULTI-SHOT REFERENCE ACQUISITION
        # ======
        print(f"\n=== Multi-shot Reference Acquisition (Slice {s + 1}) ===")

        for shot in range(acceleration_factor):
            print(f"\n--- Reference Shot {shot + 1}/{acceleration_factor} ---")

            # Inversion pulse for this shot (reuse existing rf_inv)
            rf_inv.freq_offset = gz_inv.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
            seq.add_block(rf_inv, gz_inv)
            seq.add_block(gzr_inv)
            seq.add_block(pp.make_delay(10e-3))

            # Fat sat
            seq.add_block(rf_fs, gz_fs)

            # RF excitation (using first flip angle)
            rf_ref.freq_offset = gz_ref.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
            rf180.freq_offset = gz180.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
            seq.add_block(rf_ref, gz_ref, trig)

            # TE delays and refocusing
            seq.add_block(pp.make_delay(delay_TE1))

            # For multi-shot reference, we need to adjust the pre-phasing for each shot
            # Calculate how many additional blips we need to get to this shot's starting line
            shot_indices = pe_indices_multishot[shot]
            first_line_this_shot = shot_indices[0]

            # Use the same gy_pre but add additional phase encoding to reach the shot's starting line
            additional_blips = first_line_this_shot - 1  # -1 because pe_indices start from 1

            if additional_blips > 0:
                # Create combined phase encoding gradient (gy_pre + additional)
                total_area = gy_pre.area - additional_blips * delta_k
                gy_combined = pp.make_trapezoid('y', system=system, area=total_area,
                                                duration=pp.calc_duration(gx_pre, gy_pre))
                gy_combined.amplitude = gy_combined.amplitude * pe_enable
                gy_combined.delay = pp.calc_duration(rf180)
                seq.add_block(rf180, gz180n, pp.make_delay(delay_TE2), gx_pre, gy_combined)
            else:
                seq.add_block(rf180, gz180n, pp.make_delay(delay_TE2), gx_pre, gy_pre)

            # EPI readout for this shot's k-space lines
            shot_indices = pe_indices_multishot[shot]
            for idx, i in enumerate(shot_indices):
                if idx == 0:  # First line of this shot
                    seq.add_block(gx, gy_blipup, adc)
                elif idx == len(shot_indices) - 1:  # Last line of this shot
                    seq.add_block(gx, gy_blipdown, adc)
                else:
                    seq.add_block(gx, gy_blipdownup, adc)
                gx.amplitude = -gx.amplitude

            seq.add_block(tr_spoiler)

            # Calculate timing for this reference shot
            current_ref_duration = pp.calc_duration(rf_inv, gz_inv) + pp.calc_duration(gzr_inv) + 10e-3
            current_ref_duration += pp.calc_duration(rf_fs, gz_fs)
            current_ref_duration += pp.calc_duration(rf_ref, gz_ref)
            current_ref_duration += delay_TE1
            current_ref_duration += pp.calc_duration(rf180, gz180n, gx_pre, gy_pre)  # Use gy_pre duration as reference
            current_ref_duration += len(shot_indices) * pp.calc_duration(gx, adc)
            current_ref_duration += pp.calc_duration(tr_spoiler)

            print(f"Reference shot {shot + 1}: Current duration: {current_ref_duration:.6f} s")

            # Add delay to reach desired TR (using first TR value)
            desired_tr_ref = tr_values[0]
            additional_delay_ref = desired_tr_ref - current_ref_duration

            if additional_delay_ref > 0:
                seq.add_block(pp.make_delay(additional_delay_ref))
                print(f"Adding delay of {additional_delay_ref:.6f} s to reach desired TR of {desired_tr_ref:.6f} s")
            else:
                print(
                    f"Warning: Reference shot TR ({desired_tr_ref * 1000:.1f} ms) is too short! Minimum possible TR is {current_ref_duration * 1000:.1f} ms")

        seq.add_block(pp.make_delay(3))
        # ======
        # MAIN TIME SERIES (UNCHANGED)
        # ======
        print(f"\n=== Main Time Series (Slice {s + 1}) ===")

        # Single inversion pulse for the entire time series (reuse existing rf_inv)
        rf_inv.freq_offset = gz_inv.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
        seq.add_block(rf_inv, gz_inv)
        seq.add_block(gzr_inv)
        seq.add_block(pp.make_delay(10e-3))

        # Time series loop
        for t in range(steps_number):
            seq.add_block(rf_fs, gz_fs)
            rf.freq_offset = gz.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
            rf180.freq_offset = gz180.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
            seq.add_block(rf_pulses[t], gz, trig)
            ###########################################
            # this delay_TE1 is causing the huge delay! need to fix this with the currect TE!
            seq.add_block(pp.make_delay(delay_TE1))
            ############################################
            seq.add_block(rf180, gz180n, pp.make_delay(delay_TE2), gx_pre, gy_pre)

            # EPI readout with acceleration
            for idx, i in enumerate(pe_indices_accel):
                if i == 1:
                    seq.add_block(gx, gy_blipup, adc)
                elif i == pe_indices_accel[-1]:
                    seq.add_block(gx, gy_blipdown, adc)
                else:
                    seq.add_block(gx, gy_blipdownup, adc)
                gx.amplitude = -gx.amplitude

            seq.add_block(tr_spoiler)

            current_seq_blocks_duration = pp.calc_duration(rf_pulses[t], gz)
            current_seq_blocks_duration += delay_TE1
            current_seq_blocks_duration += pp.calc_duration(rf180, gz180n, gx_pre, gy_pre)
            current_seq_blocks_duration += Ny_meas * pp.calc_duration(gx, adc)
            current_seq_blocks_duration += pp.calc_duration(tr_spoiler)
            print(f"TR {t + 1}: Total sequence duration: {current_seq_blocks_duration:.6f} s")

            desired_tr = tr_values[t]
            additional_delay = desired_tr - current_seq_blocks_duration

            if additional_delay > 0:
                seq.add_block(pp.make_delay(additional_delay))
                print(f"Adding delay of {additional_delay:.6f} s to reach desired TR of {desired_tr:.6f} s")
            else:
                print(
                    f"Warning: TR {t + 1} ({tr_values[t] * 1000:.1f} ms) is too short! Minimum possible TR is {current_seq_blocks_duration * 1000:.1f} ms")

        print(f"\n=== Completed Slice {s + 1}/{n_slices} ===\n")

    ok, error_report = seq.check_timing()
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]

    rep = seq.test_report()
    print(rep)

    if plot:
        seq.plot()
        # Calculate k-space trajectory (Python equivalent of calculateKspacePP)
        ktraj_adc, ktraj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

        # Plot the EPI k-space trajectory (equivalent to the MATLAB plot)
        plt.figure(figsize=(10, 8))
        plt.plot(ktraj[0, :], ktraj[1, :], 'b', label='Full trajectory')  # Blue line for full trajectory
        plt.plot(ktraj_adc[0, :], ktraj_adc[1, :], 'r.', label='ADC points')  # Red dots for ADC points
        plt.axis('equal')
        plt.title('EPI k-space trajectory')
        plt.xlabel('kx (1/m)')
        plt.ylabel('ky (1/m)')
        plt.legend()
        plt.grid(True)
        plt.show()

    if write_seq:
        # Calculate resolution
        resolution = fov / Nx  # meters per pixel

        # Set sequence metadata
        seq.set_definition(key='FOV', value=[fov, fov, slice_thickness])
        seq.set_definition(key='Name', value='epi_multishot_ref')
        seq.set_definition(key='Resolution', value=[resolution, resolution, slice_thickness])
        seq.set_definition(key='Ny', value=Ny)
        seq.set_definition(key='Nx', value=Nx)
        seq.set_definition(key='TimeSteps', value=steps_number)
        seq.set_definition(key='FrequencyEncodingSteps', value=adc_samples)
        seq.set_definition(key='AccelerationFactor', value=acceleration_factor)
        seq.set_definition(key='PartialFourierFactor', value=part_fourier_factor)
        seq.set_definition(key='EchoTime', value=TE)
        seq.set_definition(key='RepetitionTime', value=tr_values)
        seq.set_definition(key='FlipAngles', value=flip_angles)
        seq.set_definition(key='NySampled', value=Ny_meas)
        seq.set_definition(key='NyPre', value=Ny_pre)
        seq.set_definition(key='NyPost', value=Ny_post)
        seq.set_definition(key='MultiShotReference', value=True)
        seq.set_definition(key='ReferenceShots', value=acceleration_factor)

        seq.write(seq_filename)

    return seq


if __name__ == '__main__':
    main(plot=True, write_seq=True)