import numpy as np
from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.opts import Opts
import matplotlib.pyplot as plt


def calculate_kspace_trajectory(seq):
    """
    Calculate k-space trajectory from the sequence.
    """
    kx_points = []
    ky_points = []
    current_kx = 0
    current_ky = 0
    gamma = 42.577478518e6  # Hz/T

    # Get all blocks
    for block_counter in range(len(seq.block_events)):
        block = seq.get_block(block_counter + 1)  # Block indexing starts at 1

        # Process x gradient
        try:
            if block.gx is not None:
                current_kx += gamma * block.gx.area
        except AttributeError:
            pass

        # Process y gradient
        try:
            if block.gy is not None:
                current_ky += gamma * block.gy.area
        except AttributeError:
            pass

        # Record current k-space position
        kx_points.append(current_kx)
        ky_points.append(current_ky)

    return np.array(kx_points), np.array(ky_points)


def plot_kspace_trajectory(seq):
    """
    Plot the k-space trajectory of the sequence.
    """
    # Calculate trajectory
    kx, ky = calculate_kspace_trajectory(seq)

    # Create plot
    plt.figure(figsize=(10, 10))

    # Plot trajectory line
    plt.plot(kx, ky, 'b.-', linewidth=0.5, markersize=2, alpha=0.5)

    # Mark special points
    plt.plot(kx[0], ky[0], 'ro', label='Start', markersize=8)
    plt.plot(kx[-1], ky[-1], 'go', label='End', markersize=8)
    plt.plot(0, 0, 'k+', markersize=10, label='k-space center')

    # Add time color coding
    points = plt.scatter(kx, ky, c=np.arange(len(kx)),
                         cmap='viridis', s=2, alpha=0.7)
    plt.colorbar(points, label='Event number')

    # Labels and grid
    plt.title('K-space Trajectory')
    plt.xlabel('kx (1/m)')
    plt.ylabel('ky (1/m)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')


def create_interleaved_epi_sequence(field_of_view=256e-3, matrix_size=64,
                                    segments=4, tr=100e-3, te=60e-3,
                                    slice_thickness=3e-3):
    """
    Creates an interleaved EPI sequence.
    """
    # System limits
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130,
                  slew_unit='T/m/s', rf_ringdown_time=30e-6,
                  rf_dead_time=100e-6)

    seq = Sequence(system)

    # Calculate derived parameters
    delta_k = 1 / field_of_view
    k_width = matrix_size * delta_k
    readout_time = 640e-6  # 640µs per line
    dwell_time = readout_time / matrix_size

    # Calculate lines per segment
    lines_per_segment = matrix_size // segments
    if matrix_size % segments != 0:
        raise ValueError(f"Matrix size ({matrix_size}) must be divisible by number of segments ({segments})")

    # Create segments
    for segment in range(segments):
        # RF excitation pulse
        rf_pulse = make_sinc_pulse(flip_angle=90 * np.pi / 180,
                                   system=system,
                                   duration=3e-3,
                                   slice_thickness=slice_thickness,
                                   apodization=0.5,
                                   time_bw_product=4)

        # Create slice select gradient
        gz = make_trapezoid(channel='z',
                            system=system,
                            duration=rf_pulse.shape_dur,
                            amplitude=1.0)

        # Calculate initial phase encode value for this segment
        initial_pe = segment - (segments // 2)

        # Prephasing gradients
        pre_time = 3e-3
        gx_pre = make_trapezoid(channel='x', system=system,
                                area=-k_width / 2, duration=pre_time)

        # Calculate y prephase area based on segment
        y_area = -(initial_pe * segments * delta_k)
        gy_pre = make_trapezoid(channel='y', system=system,
                                area=y_area, duration=pre_time)

        # Add sequence blocks for this segment
        seq.add_block(rf_pulse, gz)
        seq.add_block(make_delay(1e-3))
        seq.add_block(gx_pre, gy_pre)

        # EPI readout for this segment
        for line in range(lines_per_segment):
            # Readout gradient and ADC
            gx = make_trapezoid(channel='x', system=system,
                                area=k_width,
                                duration=readout_time)

            adc = make_adc(num_samples=matrix_size,
                           duration=gx.flat_time,
                           delay=gx.rise_time)

            if line % 2 == 1:  # Alternate gradient polarity
                gx.amplitude = -gx.amplitude

            seq.add_block(gx, adc)

            # Phase encode blip (except for last line)
            if line < lines_per_segment - 1:
                gy_blip = make_trapezoid(channel='y', system=system,
                                         area=segments * delta_k,
                                         duration=1e-3)
                seq.add_block(gy_blip)

        # Add delay to achieve desired TR
        if segment < segments - 1:
            current_duration = seq.duration()[0]
            delay_time = tr - (current_duration % tr)
            if delay_time > 0:
                seq.add_block(make_delay(delay_time))

    # Check sequence validity
    seq.check_timing()
    return seq


# Example usage
if __name__ == "__main__":
    # Create sequence
    seq = create_interleaved_epi_sequence(matrix_size=64, segments=4)

    # Create a figure with two subplots
    plt.figure(figsize=(15, 5))

    # Plot sequence timing
    plt.subplot(121)
    seq.plot()
    plt.title('Sequence Timing')

    # Plot k-space trajectory
    plt.subplot(122)
    plot_kspace_trajectory(seq)
    plt.title('K-space Trajectory')

    plt.tight_layout()
    plt.show()

    # Save sequence
    seq.write("interleaved_epi_sequence.seq")