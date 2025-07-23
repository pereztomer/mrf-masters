import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# 'gtk4cairo', 'macosx', 'nbagg', 'notebook', 'qtagg', 'qtcairo', 'qt5agg', 'qt5cairo', 'tkagg', 'tkcairo', 'webagg', 'wx', 'wxagg', 'wxcairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template', 'module://backend_interagg', 'inline']
import numpy as np
import MRzeroCore as mr0
import pypulseq as pp
import torch
import math


def rf_spoiling_phase(j, phi0_deg=117):
    """RF spoiling phase: Φⱼ = ½ Φ₀ (j² + j + 2)"""
    phase = 0.5 * phi0_deg * (j ** 2 + j + 2)
    phase = phase % 360
    return phase


# Display all shots + combined results with trajectories
def display_shots(shots, x_freq_per_shot, y_freq_per_shot):
    R = len(shots)
    fig, axes = plt.subplots(3, R + 1, figsize=(3 * (R + 1), 9))

    # Individual shots
    for i, shot in enumerate(shots):
        # Row 1: K-space
        axes[0, i].imshow(np.log(np.abs(shot) + 1), cmap='gray')
        axes[0, i].set_title(f'Shot {i + 1} K-space')
        axes[0, i].axis('off')

        # Row 2: Image space
        spectrum = np.fft.fftshift(shot)
        space = np.fft.fft2(spectrum)
        space = np.fft.ifftshift(space)
        img = np.abs(space)
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f'Shot {i + 1} Image')
        axes[1, i].axis('off')

        # Row 3: Individual trajectory
        x_traj = x_freq_per_shot[i]
        y_traj = y_freq_per_shot[i]
        # Only plot non-zero points (where data exists)
        mask = np.abs(shot) > 0
        if np.any(mask):
            axes[2, i].scatter(x_traj[mask], y_traj[mask], c=range(np.sum(mask)),
                               cmap='viridis', s=1, alpha=0.7)
        axes[2, i].set_title(f'Shot {i + 1} Trajectory')
        axes[2, i].set_aspect('equal')
        axes[2, i].grid(True, alpha=0.3)

    # Combined results
    combined_kspace = np.sum(shots, axis=0)
    spectrum = np.fft.fftshift(combined_kspace)
    space = np.fft.fft2(spectrum)
    space = np.fft.ifftshift(space)
    combined_img = np.abs(space)

    # Row 1: Combined K-space
    axes[0, R].imshow(np.log(np.abs(combined_kspace) + 1), cmap='gray')
    axes[0, R].set_title('Combined K-space')
    axes[0, R].axis('off')

    # Row 2: Combined Image
    axes[1, R].imshow(combined_img, cmap='gray')
    axes[1, R].set_title('Combined Image')
    axes[1, R].axis('off')

    # Row 3: All trajectories overlaid
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
    for i in range(R):
        x_traj = x_freq_per_shot[i]
        y_traj = y_freq_per_shot[i]
        mask = np.abs(shots[i]) > 0
        if np.any(mask):
            axes[2, R].scatter(x_traj[mask], y_traj[mask],
                               c=colors[i % len(colors)], s=1, alpha=0.7,
                               label=f'Shot {i + 1}')
    axes[2, R].set_title('All Trajectories')
    axes[2, R].set_aspect('equal')
    axes[2, R].grid(True, alpha=0.3)
    axes[2, R].legend()

    plt.tight_layout()
    plt.show()


# Usage:

def display_shots_original(shots):
    R = len(shots)
    fig, axes = plt.subplots(2, R + 1, figsize=(3 * (R + 1), 6))

    # Individual shots
    for i, shot in enumerate(shots):
        # K-space
        axes[0, i].imshow(np.log(np.abs(shot) + 1), cmap='gray')
        axes[0, i].set_title(f'Shot {i + 1} K-space')
        axes[0, i].axis('off')

        # Image space
        spectrum = np.fft.fftshift(shot)
        space = np.fft.fft2(spectrum)
        space = np.fft.ifftshift(space)

        img = np.abs(space)
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f'Shot {i + 1} Image')
        axes[1, i].axis('off')

    # Combined
    combined_kspace = np.sum(shots, axis=0)
    spectrum = np.fft.fftshift(combined_kspace)
    space = np.fft.fft2(spectrum)
    space = np.fft.ifftshift(space)

    combined_img = np.abs(space)

    axes[0, R].imshow(np.log(np.abs(combined_kspace) + 1), cmap='gray')
    axes[0, R].set_title('Combined K-space')
    axes[0, R].axis('off')

    axes[1, R].imshow(combined_img, cmap='gray')
    axes[1, R].set_title('Combined Image')
    axes[1, R].axis('off')

    plt.tight_layout()
    plt.show()


def epi_gre_multi_shot():
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.rcParams['figure.dpi'] = 100  # 200 e.g. is really fine, but slower

    experiment_id = 'exB09_GRE_EPI_2D'

    # %% S1. SETUP sys

    # choose the scanner limits
    system = pp.Opts(
        max_grad=60, grad_unit='mT/m', max_slew=100, slew_unit='T/m/s',
        rf_ringdown_time=30e-6, rf_dead_time=100e-6,
        adc_dead_time=20e-6, grad_raster_time=10e-6
    )

    # %% S2. DEFINE the sequence
    seq = pp.Sequence()

    # Define FOV and resolution
    fov = 220e-3
    slice_thickness = 8e-3
    sz = (220, 220)  # spin system size / resolution
    Nread = Nphase = 192  # frequency encoding steps/samples
    R = 3
    assert Nread % 2 == 0 and Nread % R == 0
    fourier_factor = 9/16
    # fourier_factor = 1
    Nphase_in_practive = int((Nread / R) * fourier_factor)  # phase encoding steps/samples
    rf_spoiling_inc = 117  # RF spoiling increment

    # # Define rf events
    rf1, gz, _ = pp.make_sinc_pulse(
        flip_angle=15 * np.pi / 180, duration=0.2e-3,
        slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
        system=system, return_gz=True
    )


    gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, system=system)

    # Define other gradients and ADC events
    adc_duration_OG = 0.00065  # @param {type: "slider", min: 0.25e-3, max: 10e-3, step: 0.05e-3}
    a = int(system.adc_raster_time * Nread * 10 ** 7)
    b = int(system.grad_raster_time * 10 ** 7)
    c = int(adc_duration_OG * 10 ** 7)
    lcm_ab = abs(a * b) // np.gcd(a, b)
    adc_raster_duration = (lcm_ab if round(c / lcm_ab) == 0 else round(c / lcm_ab) * lcm_ab) / 10 ** 7

    # adc_raster_duration = 0.0006
    gx = pp.make_trapezoid(channel='x', flat_area=Nread / fov, flat_time=adc_raster_duration, system=system)
    gx_ = pp.make_trapezoid(channel='x', flat_area=-Nread / fov, flat_time=adc_raster_duration, system=system)
    adc = pp.make_adc(num_samples=Nread, duration=adc_raster_duration, phase_offset=0 * np.pi / 180,
                      delay=gx.rise_time, system=system)
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, system=system)

    blip_duration = 0.13e-3  # @param {type: "slider", min: 0.1e-3, max: 50e-3, step: 0.05e-3}
    gp_blip = pp.make_trapezoid(channel='y', area=(1 / fov) * R, duration=blip_duration, system=system)

    gp_shot_reference = pp.make_trapezoid(channel='y', area=(-(Nphase_in_practive * R // 2)) / fov, system=system)

    gy_spoil_reference = pp.make_trapezoid(channel='y', area=4 * (-Nphase_in_practive * R) / fov, system=system)


    TE = 50 / 1000
    TR = 170 / 1000

    seq_time_to_center = (pp.calc_duration(rf1, gz) +
                          pp.calc_duration(gx_pre, gp_shot_reference, gz_reph) +
                          Nphase_in_practive / 4 * (pp.calc_duration(gx, adc) + pp.calc_duration(gp_blip) + pp.calc_duration(gx_,
                                                                                                                 adc) + pp.calc_duration(
                gp_blip)))

    assert TE > seq_time_to_center
    additional_delay = TE - seq_time_to_center

    # gradient spoiling
    gx_spoil = pp.make_trapezoid(channel='x', area=-4 * Nread / fov, system=system)
    gz_spoil = pp.make_trapezoid(channel='z', area=20 / slice_thickness, system=system)

    seq_time_after_echo = seq_time_to_center + Nphase_in_practive / 4 * (pp.calc_duration(gx, adc) +
                                                             pp.calc_duration(gp_blip) +
                                                             pp.calc_duration(gx_,adc) + pp.calc_duration(gp_blip)) + pp.calc_duration( gx_spoil, gy_spoil_reference, gz_spoil)


    assert TR > seq_time_after_echo
    additional_delay_for_tr = TR - seq_time_after_echo

    rf_phase = 0
    rf_inc = 0

    # ======
    # CONSTRUCT SEQUENCE
    # ======
    for i in range(R):
        rf1.phase_offset = rf_phase / 180 * np.pi
        print(f"rf_phase: {rf_phase}")
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
        seq.add_block(rf1, gz)
        seq.add_block(pp.make_delay(additional_delay))
        gp_shot = pp.make_trapezoid(channel='y', area=(-(Nphase // 2) + i) / fov, system=system)
        seq.add_block(gx_pre, gp_shot, gz_reph)
        for ii in range(0, Nphase_in_practive // 2):
            seq.add_block(gx, adc)
            seq.add_block(gp_blip)
            seq.add_block(gx_, adc)
            seq.add_block(gp_blip)


        # gp_shot.amplitude = -gp_shot.amplitude
        gy_spoil = pp.make_trapezoid(channel='y', area=4 * (-Nphase_in_practive * R + i) / fov, system=system)
        seq.add_block(pp.make_delay(additional_delay_for_tr), gx_spoil, gy_spoil, gz_spoil)

    rep = seq.test_report()
    print(rep)

    ############################################
    # quick 2D brain phantom sim and plot
    signal = mr0.util.simulate_2d(seq)
    seq.plot(plot_now=False)
    mr0.util.insert_signal_plot(seq=seq, signal=signal.numpy())
    plt.show()

    kspace_frequencies = torch.Tensor(seq.calculate_kspace()[0])
    shots = []
    x_freq_per_shot = []
    y_freq_per_shot = []
    for index in range(R):
        single_shot = signal[index * Nread * Nphase_in_practive:(index + 1) * Nread * Nphase_in_practive]
        kspace_shot = kspace_frequencies[:, index * Nread * Nphase_in_practive:(index + 1) * Nread * Nphase_in_practive]
        x_freq_shot = kspace_shot[0].unsqueeze(1)
        y_freq_shot = kspace_shot[1].unsqueeze(1)

        single_shot = torch.reshape(single_shot, (Nphase_in_practive, Nread)).clone().T
        x_freq_shot = torch.reshape(x_freq_shot, (Nphase_in_practive, Nread)).clone().T
        y_freq_shot = torch.reshape(y_freq_shot, (Nphase_in_practive, Nread)).clone().T

        single_shot[:, 0::2] = torch.flip(single_shot[:, 0::2], [0])[:, :]
        x_freq_shot[:, 0::2] = torch.flip(x_freq_shot[:, 0::2], [0])[:, :]
        y_freq_shot[:, 0::2] = torch.flip(y_freq_shot[:, 0::2], [0])[:, :]

        expanded_kspace_per_shot = np.zeros((Nread, Nread), dtype=complex)
        expanded_kspace_per_shot[:, index: int(Nread * fourier_factor):R] = single_shot

        expanded_x_freq_per_shot = np.zeros((Nread, Nread), dtype=complex)
        expanded_x_freq_per_shot[:, index:int(Nread * fourier_factor):R] = x_freq_shot

        expanded_y_freq_per_shot = np.zeros((Nread, Nread), dtype=complex)
        expanded_y_freq_per_shot[:, index:int(Nread * fourier_factor):R] = y_freq_shot

        shots.append(expanded_kspace_per_shot)
        x_freq_per_shot.append(expanded_x_freq_per_shot)
        y_freq_per_shot.append(expanded_y_freq_per_shot)

    display_shots(shots, x_freq_per_shot, y_freq_per_shot)


def main():
    # run epi gre
    epi_gre_multi_shot()


if __name__ == '__main__':
    main()
