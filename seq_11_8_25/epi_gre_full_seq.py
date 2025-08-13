import numpy as np
import pypulseq as pp

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import MRzeroCore as mr0
import torch


def rf_spoiling_phase(j, phi0_deg=117):
    """RF spoiling phase: Φⱼ = ½ Φ₀ (j² + j + 2)"""
    phase = 0.5 * phi0_deg * (j ** 2 + j + 2)
    return phase % 360


def mrf_epi_sequence():
    # ======
    # SETUP
    # ======
    system = pp.Opts(
        max_grad=60, grad_unit='mT/m', max_slew=100, slew_unit='T/m/s',
        rf_ringdown_time=30e-6, rf_dead_time=100e-6,
        adc_dead_time=20e-6, grad_raster_time=10e-6
    )

    seq = pp.Sequence(system)

    # Imaging parameters
    fov = 220e-3
    slice_thickness = 8e-3
    Nread = Nphase = 36
    TE = 18 / 1000  # 18MS
    R = 3
    TI = 50
    assert Nread % 2 == 0 and Nread % R == 0

    seq_filename = f"gre_epi_{Nread}.seq"

    # Partial Fourier
    fourier_factor = 9 / 16
    # fourier_factor = 1
    Nphase_in_practice = int((Nread / R) * fourier_factor)
    rf_spoiling_inc = 117  # RF spoiling increment

    # MRF patterns
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

    # flip_angles = flip_angles[:5]
    # tr_values_ms = tr_values_ms[:5]
    tr_values = [tr / 1000 for tr in tr_values_ms]  # Convert to seconds

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

    rf180_inversion = pp.make_adiabatic_pulse(
        pulse_type='hypsec', system=system, duration=10.24e-3, dwell=1e-5, delay=system.rf_dead_time
    )

    # RF pulses for each flip angle
    rf_pulses = []
    gz_list = []
    gz_reph_list = []

    for flip in flip_angles:
        rf, gz, gz_reph = pp.make_sinc_pulse(
            flip_angle=flip * np.pi / 180,
            slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
            system=system, return_gz=True
        )
        rf_pulses.append(rf)
        gz_list.append(gz)
        gz_reph_list.append(gz_reph)

    # EPI gradients and ADC (your original approach)
    adc_duration_OG = 0.00065
    a = int(system.adc_raster_time * Nread * 10 ** 7)
    b = int(system.grad_raster_time * 10 ** 7)
    c = int(adc_duration_OG * 10 ** 7)
    lcm_ab = abs(a * b) // np.gcd(a, b)
    adc_raster_duration = (lcm_ab if round(c / lcm_ab) == 0 else round(c / lcm_ab) * lcm_ab) / 10 ** 7

    gx = pp.make_trapezoid(channel='x', flat_area=Nread / fov, flat_time=adc_raster_duration, system=system)
    gx_ = pp.make_trapezoid(channel='x', flat_area=-Nread / fov, flat_time=adc_raster_duration, system=system)
    adc = pp.make_adc(num_samples=Nread, duration=adc_raster_duration,
                      delay=gx.rise_time, system=system)

    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, system=system)

    # Phase encoding
    gp_blip = pp.make_trapezoid(channel='y', area=(1 / fov) * R, system=system)

    # Spoiling x and z directions
    gx_spoil = pp.make_trapezoid(channel='x', area=-4 * Nread / fov, system=system)
    gz_spoil = pp.make_trapezoid(channel='z', area=20 / slice_thickness, system=system)

    rf_phase = 0
    rf_inc = 0

    # ======
    # PHASE 1: MULTI-SHOT REFERENCE ACQUISITION
    # ======
    print("=== PHASE 1: Multi-shot Reference Acquisition ===")

    for i in range(R):
        print(f"Reference shot {i + 1}/{R}")
        seq.add_block(rf180_inversion)
        inversion_time = pp.calc_duration(rf180_inversion)
        fat_null_time = pp.calc_duration(rf_fs, gz_fs)
        additional_time = TI / 1000 - inversion_time - fat_null_time
        if additional_time < 0:
            print("make sure that fat nulling + inversion is shorter than 50ms")
        else:
            seq.add_block(pp.make_delay(additional_time))

        # RF spoiling
        seq.add_block(rf_fs, gz_fs)
        rf_pulses[0].phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

        # Excitation
        seq.add_block(rf_pulses[0], gz_list[0])

        # Pre-phasing for this shot
        # gp_pre = pp.make_trapezoid(channel='y', area=(-(Nphase // 2) + i) / fov, system=system)
        gp_pre = pp.make_trapezoid(channel='y', area=(-Nphase * (fourier_factor - 0.5) + i) / fov, system=system)

        # Adding delay for proper TE
        seq.add_block(gx_pre, gp_pre, gz_reph_list[0])
        # if fourier_factor == 1:
        #     # time_to_kspace_center = (((((Nphase_in_practice // 4) - 1) *
        #     #                          pp.calc_duration(gx, adc) +
        #     #                          pp.calc_duration(gx_, adc) +
        #     #                          2 * pp.calc_duration(gp_blip)) +
        #     #                          pp.calc_duration(gx, adc)) +
        #     #                          pp.calc_duration(gp_blip))
        #
        #     term_1 = pp.calc_duration(gx, adc) + pp.calc_duration(gx_, adc) + 2 * pp.calc_duration(gp_blip)
        #     term_2 = pp.calc_duration(gx, adc) + pp.calc_duration(gp_blip) + 0.5 * pp.calc_duration(gx, adc)
        #     time_to_kspace_center = ((Nphase_in_practice // 4) - 1) * term_1 + term_2
        #     if time_to_kspace_center < TE:
        #         additional_time_to_TE = TE - time_to_kspace_center
        #         seq.add_block(pp.make_delay(additional_time_to_TE))
        #     else:
        #         raise "TE is too short!"
        # else:
        #     raise NotImplementedError(f"fourier factor {fourier_factor} TE proper calculation not implemented")
        # seq.add_block(pp.make_delay(9.7 / 1000))  # custom delay only for 36X36 matrix to have echo time = 18ms
        # EPI readout (your original structure)
        for ii in range(0, Nphase_in_practice // 2):
            seq.add_block(gx, adc)
            seq.add_block(gp_blip)
            seq.add_block(gx_, adc)
            seq.add_block(gp_blip)

        # Spoiling
        gy_spoil = pp.make_trapezoid(channel='y', area=4 * (-Nphase_in_practice * R + i) / fov, system=system)

        seq.add_block(gx_spoil, gy_spoil, gz_spoil)

        # TR timing
        current_duration = (pp.calc_duration(rf_pulses[0], gz_list[0]) +
                            pp.calc_duration(gx_pre, gp_pre, gz_reph_list[0]) +
                            Nphase_in_practice // 2 * (pp.calc_duration(gx, adc) +
                                                       pp.calc_duration(gp_blip) +
                                                       pp.calc_duration(gx_, adc) +
                                                       pp.calc_duration(gp_blip)) +
                            pp.calc_duration(gx_spoil, gy_spoil))

        tr_delay = tr_values[0] - current_duration
        if tr_delay > 0:
            seq.add_block(pp.make_delay(tr_delay))

        seq.add_block(pp.make_delay(1000))

    phase1_duration = seq.duration()[0]
    # Delay between phases
    seq.add_block(pp.make_delay(1000))
    phase2_start_time = seq.duration()[0]
    delay_between_phases = phase2_start_time - phase1_duration

    # ======
    # PHASE 2: MRF TIME SERIES (SINGLE-SHOT ACCELERATED)
    # ======
    # print("=== PHASE 2: MRF Time Series ===")
    #
    # rf_phase = 0
    # rf_inc = 0
    #
    # seq.add_block(rf180_inversion)
    # inversion_time = pp.calc_duration(rf180_inversion)
    # fat_null_time = pp.calc_duration(rf_fs, gz_fs)
    # additional_time = TI / 1000 - inversion_time - fat_null_time
    # seq.add_block(rf180_inversion)
    # if additional_time < 0:
    #     print("make sure that fat nulling + inversion is shorter than 50ms")
    # else:
    #     seq.add_block(pp.make_delay(additional_time))
    #
    # for t in range(len(flip_angles)):
    #     print(f"MRF time point {t + 1}/{len(flip_angles)}: FA={flip_angles[t]}°, TR={tr_values[t] * 1000:.0f}ms")
    #     seq.add_block(rf_fs, gz_fs)
    #     # RF spoiling
    #     rf_pulses[t].phase_offset = rf_phase / 180 * np.pi
    #     adc.phase_offset = rf_phase / 180 * np.pi
    #     rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
    #     rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
    #
    #     # Excitation
    #     seq.add_block(rf_pulses[t], gz_list[t])
    #
    #     # Pre-phasing for accelerated acquisition (every R-th line)
    #     gp_pre = pp.make_trapezoid(channel='y', area=(-(Nphase // 2)) / fov, system=system)
    #     seq.add_block(gx_pre, gp_pre, gz_reph_list[t])
    #
    #     # Adding delay for proper TE
    #     # if fourier_factor == 1:
    #     #     # time_to_kspace_center = (((((Nphase_in_practice // 4) - 1) *
    #     #     #                            pp.calc_duration(gx, adc) +
    #     #     #                            pp.calc_duration(gx_, adc) +
    #     #     #                            2 * pp.calc_duration(gp_blip)) +
    #     #     #                           pp.calc_duration(gx, adc)) +
    #     #     #                          pp.calc_duration(gp_blip))
    #     #     term_1 = pp.calc_duration(gx, adc) + pp.calc_duration(gx_, adc) + 2 * pp.calc_duration(gp_blip)
    #     #     term_2 = pp.calc_duration(gx, adc) + pp.calc_duration(gp_blip) + 0.5 * pp.calc_duration(gx, adc)
    #     #     time_to_kspace_center = ((Nphase_in_practice // 4) - 1) * term_1 + term_2
    #     #     if time_to_kspace_center < TE:
    #     #         additional_time_to_TE = TE - time_to_kspace_center
    #     #         seq.add_block(pp.make_delay(additional_time_to_TE))
    #     #     else:
    #     #         raise "TE is too short!"
    #     # else:
    #     #     raise NotImplementedError(f"fourier factor {fourier_factor} TE proper calculation not implemented")
    #
    #     # seq.add_block(pp.make_delay(9.7 / 1000))  # custom delay only for 36X36 matrix to have echo time = 18ms
    #
    #     # Single-shot EPI readout (accelerated by R)
    #     for ii in range(0, Nphase_in_practice // 2):
    #         seq.add_block(gx, adc)
    #         seq.add_block(gp_blip)
    #         seq.add_block(gx_, adc)
    #         seq.add_block(gp_blip)
    #
    #     # Spoiling
    #     gy_spoil = pp.make_trapezoid(channel='y', area=4 * (-Nphase_in_practice * R) / fov, system=system)
    #     seq.add_block(gx_spoil, gy_spoil, gz_spoil)
    #
    #     # TR timing
    #     current_duration = (pp.calc_duration(rf_pulses[t], gz_list[t]) +
    #                         pp.calc_duration(gx_pre, gp_pre, gz_reph_list[t]) +
    #                         Nphase_in_practice // 2 * (pp.calc_duration(gx, adc) +
    #                                                    pp.calc_duration(gp_blip) +
    #                                                    pp.calc_duration(gx_, adc) +
    #                                                    pp.calc_duration(gp_blip)) +
    #                         pp.calc_duration(gx_spoil, gy_spoil, gz_spoil))
    #
    #     tr_delay = tr_values[t] - current_duration
    #     if tr_delay > 0:
    #         seq.add_block(pp.make_delay(tr_delay))
    #     else:
    #         print(
    #             f"Warning: TR {t + 1} too short! Need {current_duration * 1000:.1f}ms, got {tr_values[t] * 1000:.1f}ms")
    #
    # # Track end of Phase 2
    phase2_end_time = seq.duration()[0]
    phase2_duration = phase2_end_time - phase2_start_time

    # # ======
    # # FINALIZE
    # # ======
    ok, error_report = seq.check_timing()
    if ok:
        print('✅ Timing check passed')
    else:
        print('❌ Timing check failed:')
        [print(f"  - {e}") for e in error_report]

    rep = seq.test_report()
    print(rep)

    print(f"\n📊 Sequence Summary:")
    print(f"  - Phase 1: {R} reference shots, fully sampled")
    print(f"  - Phase 2: {len(flip_angles)} time points, R={R} accelerated")
    print(f"  - Partial Fourier: {fourier_factor}")
    print(f"  - Phase encoding lines per shot: {Nphase_in_practice}")
    print(f"  - Reference shots duration: {phase1_duration:.2f} seconds")
    print(f"  - Time series duration: {phase2_duration:.2f} seconds")
    print(f"  - Delay between phases: {delay_between_phases:.2f} seconds")
    print(f"  - Total duration: {seq.duration()[0]:.2f} seconds")

    # Calculate resolution
    resolution = fov / Nphase  # meters per pixel

    # Set sequence metadata
    seq.set_definition(key='FOV', value=[fov, fov, slice_thickness])
    seq.set_definition(key='Name', value='epi_multishot_ref')
    seq.set_definition(key='Resolution', value=[resolution, resolution, slice_thickness])
    seq.set_definition(key='Ny', value=Nread)
    seq.set_definition(key='Nx', value=Nphase)
    seq.set_definition(key='TimeSteps', value=len(flip_angles))
    seq.set_definition(key='AccelerationFactor', value=R)
    seq.set_definition(key='PartialFourierFactor', value=fourier_factor)
    seq.set_definition(key='EchoTime', value=-1)
    seq.set_definition(key='RepetitionTime', value=tr_values)
    seq.set_definition(key='FlipAngles', value=flip_angles)
    seq.set_definition(key='NySampled', value=Nphase_in_practice)
    seq.set_definition(key='NyPre', value=-1)
    seq.set_definition(key='NyPost', value=-1)
    seq.set_definition(key='MultiShotReference', value=True)
    seq.set_definition(key='ReferenceShots', value=R)

    seq.write(seq_filename)


# Run the sequence
if __name__ == '__main__':
    sequence = mrf_epi_sequence()
