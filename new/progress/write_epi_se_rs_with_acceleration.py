import math
from datetime import date

import numpy as np
import pypulseq as pp

# =============================================================================
# Single‑shot EPI sequence with optional in‑plane acceleration R
# =============================================================================
# * R = 1  – fully‑sampled EPI (identical timing to your original script).
# * R ≥ 2 – undersampling by a factor R in the phase‑encode (k_y) direction.
#           The blips, pre‑phasers and TE delays are automatically adjusted so
#           that **the echo time (TE) stays exactly the value specified below**
#           regardless of R or of the partial‑Fourier factor.
# -----------------------------------------------------------------------------


def main(*, plot: bool = False, write_seq: bool = False, R: int = 1,
         seq_filename: str | None = None):
    """Generate a Siemens‑style .seq file for an EPI trajectory.

    Parameters
    ----------
    plot        : bool, optional
        Show timing diagram if *True*.
    write_seq   : bool, optional
        Write the sequence to disk if *True*.
    R           : int, optional
        In‑plane acceleration factor (GRAPPA/SENSE).  ``R ≥ 1``.
    seq_filename: str | None, optional
        Output filename (``.seq``).  If *None*, a descriptive name is created.
    """

    # =====================  USER‑ADJUSTABLE PARAMETERS  =====================
    fov = 220e-3            # In‑plane FOV [m]
    Nx  = 192               # Read‑out samples (matrix size in k_x)
    Ny  = 192               # Logical phase‑encode lines (full matrix)

    slice_thickness = 3e-3  # Slice thickness [m]
    n_slices        = 1     # Single‑shot per slice

    TE  = 0.200             # ***Echo time [s] – kept constant!***

    readout_time = 2 * 4.2e-4   # Total ADC window [s]
    ro_os        = 1            # Receiver oversampling factor

    part_fourier_factor = 1.0   # 1.0 = full k‑space, 0.5 = half Fourier

    # RF pulses --------------------------------------------------------------
    t_RF_ex   = 2e-3        # Excitation sinc duration [s]
    t_RF_ref  = 2e-3        # 180° refocusing sinc duration [s]
    flip_angles  = [15.0]   # One excitation per TR (single‑shot)
    tr_values_ms = [200]    # Nominal TR list (ms)

    # Spoiler scaling factors -----------------------------------------------
    spoil_factor_slice = 1.5   # Around 180° pulse
    spoil_factor_tr    = 2.0   # At end of TR

    # ======================  PARAMETER CHECKS  ==============================
    assert len(flip_angles) == len(tr_values_ms), "flip‑angle/TR mismatch"
    assert R >= 1, "Acceleration factor R must be ≥ 1"
    assert part_fourier_factor >= 0.5, "Partial Fourier must be ≥ 0.5"

    n_reps = len(flip_angles)

    # ============================  SYSTEM LIMITS  ===========================
    system = pp.Opts(max_grad=60, grad_unit='mT/m',
                     max_slew=150, slew_unit='T/m/s',
                     rf_ringdown_time=30e-6, rf_dead_time=100e-6)

    seq = pp.Sequence(system)

    # ======================  K‑SPACE METRICS  ===============================
    delta_k = 1 / fov
    k_width = Nx * delta_k

    Ny_pre  = round(Ny/2 - 1)                               # lines before ky=0 (excl.)
    Ny_post = max(1, round(part_fourier_factor*Ny - Ny_pre))# centre + after
    Ny_meas = Ny_pre + Ny_post                              # logical lines measured
    Ny_acq  = math.ceil(Ny_meas / R)                        # physical echoes acquired

    print(f"Partial Fourier = {part_fourier_factor:g}")
    print(f"Acceleration R  = {R:d} → acquire {Ny_acq} / {Ny_meas} echoes")

    # =======================  FAT‑SAT PULSE  ================================
    B0 = 2.89  # Tesla
    sat_ppm  = -3.45
    sat_freq = sat_ppm * 1e-6 * B0 * system.gamma

    rf_fs = pp.make_gauss_pulse(flip_angle=110*np.pi/180, system=system,
                                duration=8e-3, bandwidth=abs(sat_freq),
                                freq_offset=sat_freq, delay=system.rf_dead_time)
    gz_fs = pp.make_trapezoid('z', system=system,
                              delay=pp.calc_duration(rf_fs), area=1/1e-4)

    # ======================  EXCITATION PULSES  =============================
    rf_ex_list = []  # store tuples (rf, gz, gz_reph)
    for fa in flip_angles:
        rf_ex, gz_ex, gz_reph = pp.make_sinc_pulse(flip_angle=fa*np.pi/180,
                                                   system=system,
                                                   duration=t_RF_ex,
                                                   slice_thickness=slice_thickness,
                                                   apodization=0.5,
                                                   time_bw_product=4,
                                                   return_gz=True,
                                                   delay=system.rf_dead_time)
        rf_ex_list.append((rf_ex, gz_ex, gz_reph))

    # ===================  180° REFOCUSING & SPOILERS  =======================
    rf180, gz180, _ = pp.make_sinc_pulse(flip_angle=np.pi, system=system,
                                         duration=t_RF_ref,
                                         slice_thickness=slice_thickness,
                                         apodization=0.5, time_bw_product=4,
                                         phase_offset=np.pi/2, use='refocusing',
                                         return_gz=True,
                                         delay=system.rf_dead_time)
    # Slice spoilers around 180°
    _, gzr1_t, gzr1_a = pp.make_extended_trapezoid_area('z', system=system,
                                                        grad_start=0,
                                                        grad_end=gz180.amplitude,
                                                        area=spoil_factor_slice*rf_ex_list[0][1].area)
    _, gzr2_t, gzr2_a = pp.make_extended_trapezoid_area('z', system=system,
                                                        grad_start=gz180.amplitude,
                                                        grad_end=0,
                                                        area=-rf_ex_list[0][2].area + spoil_factor_slice*rf_ex_list[0][1].area)
    if gz180.delay > (gzr1_t[3] - gz180.rise_time):
        gz180.delay -= gzr1_t[3] - gz180.rise_time
    else:
        rf180.delay += (gzr1_t[3] - gz180.rise_time) - gz180.delay

    gz180n = pp.make_extended_trapezoid('z', system=system,
                                        times=np.array([*gzr1_t,
                                                        *gzr1_t[3] + gz180.flat_time + gzr2_t]) + gz180.delay,
                                        amplitudes=np.array([*gzr1_a, *gzr2_a]))

    # =========================  END‑OF‑TR SPOILER  ==========================
    tr_spoiler = pp.make_trapezoid('z', system=system,
                                   area=spoil_factor_tr*rf_ex_list[0][1].area,
                                   duration=3e-3)

    # ======================  PHASE‑ENCODE BLIP  =============================
    blip_duration = np.ceil(2*np.sqrt(R*delta_k/system.max_slew)/(10e-6*2))*10e-6*2
    gy_blip = pp.make_trapezoid('y', system=system,
                                area=-R*delta_k, duration=blip_duration)

    # =======================  READ‑OUT GRADIENT  ============================
    extra_area = blip_duration**2/4 * system.max_slew
    gx = pp.make_trapezoid('x', system=system,
                           area=k_width + extra_area,
                           duration=readout_time + blip_duration)
    actual_area = gx.area - gx.amplitude/gx.rise_time*blip_duration**2/8 - gx.amplitude/gx.fall_time*blip_duration**2/8
    gx.amplitude *= k_width/actual_area
    gx.area = gx.amplitude*(gx.flat_time + gx.rise_time/2 + gx.fall_time/2)

    # ============================  ADC EVENT  ===============================
    adc_dwell_nyq = delta_k/gx.amplitude/ro_os
    adc_dwell     = math.floor(adc_dwell_nyq*1e7)*1e-7
    adc_samples   = math.floor(readout_time/adc_dwell/4)*4
    adc = pp.make_adc(num_samples=adc_samples, dwell=adc_dwell, delay=blip_duration/2)
    time_to_ctr = adc_dwell*((adc_samples-1)/2 + 0.5)
    adc.delay = round((gx.rise_time + gx.flat_time/2 - time_to_ctr)*1e6)*1e-6

    # Split blip for combination
    gy_up_part, gy_dn_part = pp.split_gradient_at(gy_blip, blip_duration/2, system)
    gy_up, gy_dn, _ = pp.align(right=gy_up_part, left=[gy_dn_part, gx])
    gy_dnup = pp.add_gradients((gy_dn, gy_up), system=system)

    # =========================  PRE‑PHASERS  ================================
    gx_pre = pp.make_trapezoid('x', system=system, area=-gx.area/2)
    gy_pre = pp.make_trapezoid('y', system=system, area=Ny_pre*delta_k)
    gx_pre, gy_pre = pp.align(right=gx_pre, left=gy_pre)
    gy_pre = pp.make_trapezoid('y', system=system, area=gy_pre.area,
                               duration=pp.calc_duration(gx_pre, gy_pre))

    # ==========================  TE DELAYS  =================================
    echo_spacing       = pp.calc_duration(gx)
    echo_to_center     = math.ceil(Ny_pre / R)          # echoes until ky=0
    duration_to_center = (echo_to_center - 0.5)*echo_spacing

    rf_ex_center_delay = rf_ex_list[0][0].delay + pp.calc_rf_center(rf_ex_list[0][0])[0]
    rf180_center_delay = rf180.delay + pp.calc_rf_center(rf180)[0]

    delay_TE1 = math.ceil((TE/2 - pp.calc_duration(rf_ex_list[0][0], rf_ex_list[0][1])
                           + rf_ex_center_delay - rf180_center_delay)/system.grad_raster_time)*system.grad_raster_time

    delay_TE2 = math.ceil((TE/2 - pp.calc_duration(rf180, gz180n)
                           + rf180_center_delay - duration_to_center)/system.grad_raster_time)*system.grad_raster_time
    delay_TE2 += pp.calc_duration(rf180, gz180n)

    # Place pre‑phasers relative to TE2 block
    gx_pre.delay = delay_TE2 - pp.calc_duration(gx_pre)
    gy_pre.delay = pp.calc_duration(rf180)

    # =======================  LOOP OVER SLICES & REPS =======================
    for s in range(n_slices):
        # Optional inversion could be added here if desired

        for rep in range(n_reps):
            rf_ex, gz_ex, gz_reph = rf_ex_list[rep]
            # Slice offsets (single slice => 0)
            rf_ex.freq_offset = gz_ex.amplitude*slice_thickness*(s - (n_slices-1)/2)
            rf180.freq_offset = gz180.amplitude*slice_thickness*(s - (n_slices-1)/2)

            # --- Fat sat
            seq.add_block(rf_fs, gz_fs)

            # --- Excitation
            seq.add_block(rf_ex, gz_ex)

            # --- First TE delay
            seq.add_block(pp.make_delay(delay_TE1))

            # --- 180° + TE2 + pre‑phasers
            seq.add_block(rf180, gz180n, pp.make_delay(delay_TE2), gx_pre, gy_pre)

            # --- EPI readout ------------------------------------------------
            for echo in range(1, Ny_acq+1):
                if echo == 1:
                    seq.add_block(gx, gy_up, adc)
                elif echo == Ny_acq:
                    seq.add_block(gx, gy_dn, adc)
                else:
                    seq.add_block(gx, gy_dnup, adc)
                gx.amplitude = -gx.amplitude  # read‑out polarity toggle

            # --- End‑of‑TR spoiler
            seq.add_block(tr_spoiler)

            # --- Additional delay to reach nominal TR
            blocks_dur = (pp.calc_duration(rf_ex, gz_ex) + delay_TE1 +
                           pp.calc_duration(rf180, gz180n, gx_pre, gy_pre) +
                           Ny_acq*echo_spacing + pp.calc_duration(tr_spoiler))
            desired_tr = tr_values_ms[rep]/1000.0
            if desired_tr > blocks_dur:
                seq.add_block(pp.make_delay(desired_tr - blocks_dur))

    # =====================  TIMING CHECK & REPORT  ==========================
    ok, errs = seq.check_timing()
    if not ok:
        print("Timing errors detected:")
        for e in errs:
            print(e)
        raise RuntimeError("Sequence timing invalid")
    else:
        print("Timing check passed ✓")

    print(seq.test_report())

    # =========================  VISUALISATION  ==============================
    if plot:
        seq.plot()

    # ===========================  WRITE .SEQ  ===============================
    if write_seq:
        if seq_filename is None:
            today = date.today().isoformat()
            pf_tag = int(part_fourier_factor*10)
            seq_filename = f"sequences/epi_{today}_Nx{Nx}_Ny{Ny}_PF{pf_tag}_R{R}.seq"
        seq.set_definition('FOV', [fov, fov, slice_thickness])
        seq.set_definition('Name', 'epi')
        seq.write(seq_filename)
        print(f"Sequence written to {seq_filename}")

    return seq


# =====================================================================
# Stand‑alone execution helper
# =====================================================================
if __name__ == "__main__":
    # Example call – change R or other arguments as desired
    main(plot=True, write_seq=False, R=4)
